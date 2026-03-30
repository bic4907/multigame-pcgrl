"""
RunPod Multi-Process Deployment Script

Features:
- Multi-pod parallel execution with real-time dashboard
- Emergency cleanup mechanism for interrupted execution
- Signal handlers for SIGINT (Ctrl+C) and SIGTERM
- Automatic pod termination on script exit
- Activity logging with status tracking

Safety mechanisms:
1. All created pods are registered in a global tracker
2. Signal handlers catch Ctrl+C and kill signals
3. atexit ensures cleanup on normal termination
4. Each pod process unregisters on successful cleanup
5. Emergency cleanup terminates all tracked pods on failure
"""

import os
import copy
import time
import yaml
import argparse
import signal
import atexit
from enum import Enum
from pathlib import Path
from itertools import product
from multiprocessing import Process, Manager
from multiprocessing.managers import DictProxy
from typing import Optional
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.layout import Layout

import runpod
import requests
from dotenv import load_dotenv


_active_pods: Optional[DictProxy] = None   # will be set to Manager().dict() in run_multiple_pods
_api_key: Optional[str] = None             # stored globally so signal handlers can use it
_console = Console()

config_folder_path = os.path.join(os.path.dirname(__file__), 'config')

class CmdType(Enum):
    TRAIN = "TRAIN"
    STOP = "STOP"
    WANDB_LOGIN = "WANDB_LOGIN"


def parse_args():
    """
    어떤 YAML 설정 파일을 사용할지에 대해서만 인자를 받는다.
    실제 pod 설정(name, gpu 등)은 전부 YAML에서 읽는다.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        nargs="+",
        help=(
            "YAML config file(s). "
            "Names are looked up under runpod/config, paths are used as-is."
        ),
    )
    args = parser.parse_args()

    return args


template_folder_path = os.path.join(os.path.dirname(__file__), 'template')


def expand_and_save_sweep(template_path: Path) -> Path:
    """Expand a template YAML with a ``sweep`` section into individual configs.

    For each combination of sweep parameter values (cartesian product),
    an individual YAML is written to ``config/<template_stem>/<name>.yaml``.
    Placeholders ``{key}`` in *name* and *runtime.cmds* are substituted.

    Args:
        template_path: Path to the template YAML (must contain a ``sweep`` key).

    Returns:
        Path to the generated config folder (``config/<template_stem>``).
    """
    with open(template_path, "r") as f:
        config_data = yaml.safe_load(f)

    sweep = config_data.get('sweep')
    if not sweep:
        raise ValueError(f"Template {template_path} has no 'sweep' section")

    sweep_name = template_path.stem  # e.g. "cpcgrl_reward_enum"
    output_dir = Path(config_folder_path) / sweep_name
    output_dir.mkdir(parents=True, exist_ok=True)

    keys = list(sweep.keys())
    value_lists = [sweep[k] if isinstance(sweep[k], list) else [sweep[k]] for k in keys]

    generated = []
    for combo in product(*value_lists):
        mapping = dict(zip(keys, combo))
        cfg = copy.deepcopy(config_data)
        cfg.pop('sweep', None)

        # Remove template-only comments (they are lost by yaml round-trip anyway)
        if 'name' in cfg and isinstance(cfg['name'], str):
            cfg['name'] = cfg['name'].format(**mapping)

        if 'runtime' in cfg and 'cmds' in cfg['runtime']:
            cfg['runtime']['cmds'] = [
                cmd.format(**mapping) for cmd in cfg['runtime']['cmds']
            ]

        out_file = output_dir / f"{cfg['name']}.yaml"
        with open(out_file, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        generated.append(out_file)

    _console.print(
        f"[green]✅ Generated {len(generated)} config(s) "
        f"from template '{sweep_name}' → {output_dir}[/green]"
    )
    return output_dir


def _load_configs_from_dir(dir_path: Path) -> list:
    """Load all YAML configs in *dir_path* and return a list of pod-configs."""
    yaml_files = sorted(dir_path.glob("*.yaml"))
    if not yaml_files:
        raise FileNotFoundError(f"No YAML files found in {dir_path}")

    pod_configs = []
    for yf in yaml_files:
        with open(yf, "r") as f:
            data = yaml.safe_load(f)
        pod_configs.append(create_pod_config_from_yaml(data))
    return pod_configs


def create_pod_config_from_yaml(config_data):
    """Create pod configuration from yaml data"""
    pod_cfg = config_data.get('pod', {})

    cmd = []
    for entry in config_data.get('runtime', {}).get('cmds', []):
        cmd.append(entry)

    cmd_str = " && ".join(cmd) if cmd else ""

    # 필수 필드는 없으면 명시적으로 에러를 던져서 YAML을 바로 수정하게 만든다.
    name = config_data.get('name')
    template_id = pod_cfg.get('template_id')
    gpu = pod_cfg.get('gpu')
    gpu_count = pod_cfg.get('gpu_count')

    missing = []
    if name is None:
        missing.append("name")
    if template_id is None:
        missing.append("pod.template_id")
    if gpu is None:
        missing.append("pod.gpu")
    if gpu_count is None:
        missing.append("pod.gpu_count")
    if missing:
        raise ValueError(f"Missing required fields in config: {', '.join(missing)}")

    # RunPod API configuration
    runpod_config = {
        'name': name,
        'templateId': template_id,
        'gpuTypeIds': [gpu],
        'gpuCount': gpu_count,
    }

    if pod_cfg.get('network_volume_id'):
        runpod_config['networkVolumeId'] = pod_cfg['network_volume_id']

    if pod_cfg.get('spot'):
        runpod_config['interruptible'] = True

    if cmd_str:
        runpod_config['dockerStartCmd'] = ['bash', '-lc', cmd_str]

    # Internal configuration for monitoring and termination
    options_cfg = config_data.get('options', {})
    internal_config = {
        'timeout': options_cfg.get('timeout', False),
        'time_limit': options_cfg.get('time_limit', 7200),
        'terminate': options_cfg.get('terminate', True),
    }

    # Combine configurations
    combined_config = {**runpod_config, **internal_config}

    return combined_config


def cleanup_all_pods(signum=None, frame=None):
    """Emergency cleanup function to terminate all active pods.

    Works with both the shared ``Manager().dict()`` (_active_pods) populated
    by child processes AND the global ``_api_key`` set in the parent process.
    """
    global _active_pods, _api_key

    if not _active_pods:
        return

    # snapshot: Manager proxy → regular dict so we don't hold the proxy lock
    try:
        snapshot = dict(_active_pods)
    except Exception:
        snapshot = {}

    if not snapshot:
        return

    _console.print("\n[bold red]🚨 Emergency cleanup initiated...[/bold red]")
    _console.print(f"[yellow]Terminating {len(snapshot)} active pod(s)...[/yellow]")

    # Determine the API key to use
    api_key = _api_key or os.getenv('RUNPOD_API_KEY')
    if api_key:
        runpod.api_key = api_key

    cleanup_results = []
    for pod_id in snapshot:
        try:
            pod_info = runpod.get_pod(pod_id=pod_id)
            if pod_info:
                runpod.terminate_pod(pod_id=pod_id)
                cleanup_results.append((pod_id, 'success'))
                _console.print(f"[green]✓[/green] Terminated pod: {pod_id[:12]}...")
            else:
                cleanup_results.append((pod_id, 'already_gone'))
                _console.print(f"[dim]○[/dim] Pod already gone: {pod_id[:12]}...")
        except Exception as e:
            cleanup_results.append((pod_id, f'error: {str(e)[:30]}'))
            _console.print(f"[red]✗[/red] Failed to terminate {pod_id[:12]}...: {str(e)[:50]}")

    try:
        _active_pods.clear()
    except Exception:
        pass

    success_count = sum(1 for _, status in cleanup_results if status == 'success')
    _console.print(f"\n[cyan]Cleanup complete: {success_count}/{len(cleanup_results)} pods terminated[/cyan]")

    # If called by signal, exit
    if signum is not None:
        _console.print("[yellow]Exiting...[/yellow]")
        os._exit(0)


def register_pod(pod_id, api_key=None):
    """Register a pod for cleanup tracking (uses shared Manager dict)"""
    global _active_pods
    if _active_pods is not None:
        _active_pods[pod_id] = True


def unregister_pod(pod_id):
    """Unregister a pod from cleanup tracking"""
    global _active_pods
    if _active_pods is not None:
        try:
            _active_pods.pop(pod_id, None)
        except Exception:
            pass


def run_single_pod(pod_config, status_dict, pod_id_key, API_KEY, log_queue, active_pods):
    """Run a single pod and update its status"""
    global _active_pods
    _active_pods = active_pods  # share the Manager dict into this child process
    runpod.api_key = API_KEY
    headers = {
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json'
    }
    base_url = 'https://rest.runpod.io/v1'
    
    pod_id = ''
    spinners = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
    spinner_idx = 0

    def update_status(key, **kwargs):
        """Helper function to update status dict with proper syncing"""
        current = dict(status_dict.get(key, {}))

        # Log status changes
        old_status = current.get('status', '')
        new_status = kwargs.get('status', old_status)

        current.update(kwargs)
        status_dict[key] = current

        # Add log entry if status changed (excluding spinner updates)
        if 'status' in kwargs and old_status and new_status:
            # Remove spinner characters for comparison
            old_clean = old_status.split(' ', 1)[-1] if ' ' in old_status else old_status
            new_clean = new_status.split(' ', 1)[-1] if ' ' in new_status else new_status

            if old_clean != new_clean:
                timestamp = datetime.now().strftime('%H:%M:%S')
                pod_name = current.get('name', f'Pod{key}')
                log_msg = f"[{timestamp}] {pod_name}: {old_clean} → {new_clean}"
                if 'progress' in kwargs:
                    log_msg += f" ({kwargs['progress']})"
                log_queue.append(log_msg)

    try:
        hash_str = os.urandom(4).hex()
        date_str = datetime.now().strftime('%Y%m%d-%H%M%S')
        cfg_name = pod_config.get('name', 'pod')
        pod_name = f"{cfg_name}-{hash_str}-{date_str}"

        # Update status - initialize immediately with the pod name
        update_status(pod_id_key,
            status=f'{spinners[spinner_idx % len(spinners)]} CREATING',
            pod_id='',
            name=pod_name,
            start_time=datetime.now().strftime('%H:%M:%S'),
            runtime='0s',
            error='',
            progress='Requesting pod...'
        )
        spinner_idx += 1

        update_status(pod_id_key, progress='Sending request to RunPod API...')

        # RunPod API 요청용 설정 — pod_config (YAML) 값 사용
        runpod_request = {
            'name': pod_name,
            'templateId': pod_config['templateId'],
            'gpuTypeIds': pod_config['gpuTypeIds'],
            'gpuCount': pod_config['gpuCount'],
        }

        # networkVolumeId가 있으면 추가
        if pod_config.get('networkVolumeId'):
            runpod_request['networkVolumeId'] = pod_config['networkVolumeId']

        # spot(interruptible)이면 추가
        if pod_config.get('interruptible'):
            runpod_request['interruptible'] = True

        # dockerStartCmd가 있으면 추가
        if 'dockerStartCmd' in pod_config:
            runpod_request['dockerStartCmd'] = pod_config['dockerStartCmd']

        # and stop pod
        if 'dockerStartCmd' in runpod_request:
            runpod_request['dockerStartCmd'][-1] += ' && runpodctl stop pod $RUNPOD_POD_ID'

        # ── Pod 생성 요청 (GPU 미가용 시 자동 재시도) ──
        import json as _json
        RETRY_INTERVAL = 5          # 재시도 간격(초)
        # 이 문자열 중 하나라도 에러 메시지에 포함되면 일시적 부족으로 간주
        _RETRYABLE_KEYWORDS = [
            'no instances currently',
            'no available instances',
            'insufficient capacity',
            'out of stock',
            'no gpu',
        ]

        retry_count = 0
        while True:
            _ts = datetime.now().strftime('%H:%M:%S')
            _req_summary = {k: v for k, v in runpod_request.items() if k != 'dockerStartCmd'}
            if retry_count == 0:
                log_queue.append(f"[{_ts}] {pod_name} → POST /pods  {_json.dumps(_req_summary, ensure_ascii=False)}")

            response = requests.post(f'{base_url}/pods', headers=headers, json=runpod_request)

            # ── HTTP 응답 로깅 ──
            _ts = datetime.now().strftime('%H:%M:%S')
            try:
                _resp_body = response.json()
                _resp_text = _json.dumps(_resp_body, ensure_ascii=False)
            except Exception:
                _resp_text = response.text[:300] if response.text else '(empty body)'
            log_queue.append(f"[{_ts}] {pod_name} ← HTTP {response.status_code}  {_resp_text[:300]}")

            # ── 성공 ──
            if response.ok:
                break

            # ── 실패: 재시도 가능 여부 판단 ──
            status_code = response.status_code
            try:
                resp_body = response.json()
                error_msg = resp_body.get('error', resp_body.get('message', response.text[:200]))
            except Exception:
                error_msg = response.text[:200] if response.text else 'No response body'

            error_lower = str(error_msg).lower()
            is_retryable = any(kw in error_lower for kw in _RETRYABLE_KEYWORDS)

            if not is_retryable:
                # 재시도 불가능한 에러 → 즉시 실패
                error_detail = f"[HTTP {status_code}] {error_msg}"
                update_status(pod_id_key,
                    status='❌ ERROR',
                    error=error_detail,
                    progress=f'API failed ({status_code})'
                )
                return

            # ── 재시도 가능 → 대기 후 다시 시도 ──
            retry_count += 1
            spinner_char = spinners[spinner_idx % len(spinners)]
            spinner_idx += 1
            elapsed_wait = retry_count * RETRY_INTERVAL
            elapsed_min = elapsed_wait // 60
            if elapsed_min > 0:
                wait_str = f'{elapsed_min}m{elapsed_wait % 60}s'
            else:
                wait_str = f'{elapsed_wait}s'

            update_status(pod_id_key,
                status=f'{spinner_char} WAITING',
                progress=f'No GPU available, retry #{retry_count} ({wait_str})',
                error=str(error_msg)[:200]
            )

            if retry_count % 12 == 1:  # 1분에 한 번 로그
                log_queue.append(
                    f"[{_ts}] {pod_name} ⏳ GPU unavailable, retrying every {RETRY_INTERVAL}s "
                    f"(attempt #{retry_count}, waited {wait_str})"
                )

            time.sleep(RETRY_INTERVAL)

        pod_id = response.json().get('id', '')
        if not pod_id:
            update_status(pod_id_key,
                status='❌ ERROR',
                error='No pod id returned',
                progress='Failed'
            )
            return

        if retry_count > 0:
            _ts = datetime.now().strftime('%H:%M:%S')
            log_queue.append(f"[{_ts}] {pod_name} ✅ GPU acquired after {retry_count} retries")

        # Register pod for emergency cleanup
        register_pod(pod_id, API_KEY)

        update_status(pod_id_key,
            pod_id=pod_id,
            status=f'{spinners[spinner_idx % len(spinners)]} INITIALIZING',
            progress=f'Pod created: {pod_id[:8]}...'
        )
        spinner_idx += 1

        start_time = time.time()
        timeout = pod_config.get('timeout', False)
        time_limit = pod_config.get('time_limit', 3600)
        terminate = pod_config.get('terminate', True)
        
        # Monitor pod with spinner (Pod 생성은 보통 5분 정도 소요)
        # 스피너는 빠르게(0.25s), API 폴링은 느리게(3s)
        poll_interval = 3.0
        spinner_interval = 0.25
        last_poll = 0.0
        last_spinner = 0.0
        last_pod_info = None

        while True:
            now = time.time()
            elapsed = int(now - start_time)
            elapsed_min = elapsed // 60
            elapsed_sec = elapsed % 60
            runtime_str = f'{elapsed_min}m{elapsed_sec}s' if elapsed_min > 0 else f'{elapsed}s'

            # API 폴링은 주기적으로만 호출
            if now - last_poll >= poll_interval:
                last_poll = now
                pod_info = runpod.get_pod(pod_id=pod_id)
                if not pod_info:
                    update_status(pod_id_key,
                        status='💀 TERMINATED',
                        progress='Pod not found',
                        runtime=runtime_str
                    )
                    break
                last_pod_info = pod_info

                current_status = pod_info.get('desiredStatus', 'UNKNOWN')

                # Update progress based on actual pod status with time indicators
                if current_status == 'RUNNING':
                    status_label = 'RUNNING'
                    progress_msg = 'Training in progress...'
                elif current_status == 'EXITED':
                    update_status(pod_id_key,
                        status='✅ EXITED',
                        progress='Completed',
                        runtime=runtime_str
                    )
                    break
                elif current_status == 'CREATED':
                    status_label = 'STARTING'
                    # Pod 생성은 보통 5분 걸림
                    if elapsed < 60:
                        progress_msg = 'Starting... (usually takes ~5min)'
                    elif elapsed < 180:
                        progress_msg = f'Starting... ({elapsed_min}m elapsed)'
                    elif elapsed < 300:
                        progress_msg = f'Still starting... ({elapsed_min}m, normal)'
                    else:
                        progress_msg = f'Starting... ({elapsed_min}m, taking longer)'
                else:
                    status_label = current_status
                    progress_msg = f'Status: {current_status}'

            else:
                # 폴링하지 않는 사이에는 마지막 알려진 상태 라벨을 유지
                if last_pod_info:
                    current_status = last_pod_info.get('desiredStatus', 'UNKNOWN')
                    if current_status == 'RUNNING':
                        status_label = 'RUNNING'
                    elif current_status == 'CREATED':
                        status_label = 'STARTING'
                    else:
                        status_label = current_status
                else:
                    status_label = 'CREATING'
                progress_msg = None  # Don't update progress between polls

            # 스피너는 별도로 빠르게 업데이트
            if now - last_spinner >= spinner_interval:
                last_spinner = now
                spinner_char = spinners[spinner_idx % len(spinners)]
                spinner_idx += 1

                # RUNNING이면 🚀 사용, 그 외에는 스피너 사용
                if last_pod_info and last_pod_info.get('desiredStatus') == 'RUNNING':
                    status_str = '🚀 RUNNING'
                else:
                    status_str = f'{spinner_char} {status_label}'

                # 업데이트할 정보 준비
                update_dict = {
                    'status': status_str,
                    'runtime': runtime_str
                }
                if progress_msg is not None:
                    update_dict['progress'] = progress_msg

                update_status(pod_id_key, **update_dict)

            # 타임아웃 체크
            if timeout and elapsed > time_limit:
                update_status(pod_id_key,
                    status='⏱️ TIMEOUT',
                    progress=f'Exceeded {time_limit}s limit',
                    runtime=runtime_str
                )
                break
            
            time.sleep(0.1)  # 짧게 쉬어가며 스피너를 부드럽게 업데이트

        # Cleanup
        if terminate and pod_id:
            update_status(pod_id_key, progress='Terminating...')
            pod_info = runpod.get_pod(pod_id=pod_id)
            if pod_info:
                runpod.terminate_pod(pod_id=pod_id)
                time.sleep(1)
                update_status(pod_id_key,
                    status='🛑 TERMINATED',
                    progress='Cleaned up'
                )
            # Unregister from cleanup tracking
            unregister_pod(pod_id)

    except requests.exceptions.HTTPError as e:
        # requests 라이브러리가 raise한 HTTP 에러 (raise_for_status 등)
        resp = e.response
        if resp is not None:
            try:
                body = resp.json()
                detail = body.get('error', body.get('message', resp.text[:200]))
            except Exception:
                detail = resp.text[:200] if resp.text else str(e)
            error_detail = f"[HTTP {resp.status_code}] {detail}"
        else:
            error_detail = str(e)[:200]

        update_status(pod_id_key,
            status='❌ ERROR',
            error=error_detail,
            progress='Failed with HTTP error'
        )
        if pod_id:
            unregister_pod(pod_id)

    except Exception as e:
        update_status(pod_id_key,
            status='❌ ERROR',
            error=str(e)[:200],
            progress='Failed with exception'
        )
        # Ensure pod is unregistered even on error
        if pod_id:
            unregister_pod(pod_id)


def generate_dashboard(status_dict, log_queue, max_logs=30):
    """Generate rich dashboard with status table and log panel"""
    # Create status table
    table = Table(title="RunPod Multi-Process Dashboard", show_header=True, header_style="bold magenta")
    table.add_column("Index", style="cyan", width=6)
    table.add_column("Name", style="green", width=20)
    table.add_column("Pod ID", style="yellow", width=15)
    table.add_column("Status", style="white", width=15)
    table.add_column("Progress", style="magenta", width=25)
    table.add_column("Runtime", style="blue", width=8)
    table.add_column("Start", style="white", width=8)
    table.add_column("Error", style="red", width=60)

    for idx in sorted(status_dict.keys()):
        info = status_dict[idx]
        status = info['status']
        
        # Status is already emoji-decorated from run_single_pod
        status_colored = status

        table.add_row(
            str(idx),
            info['name'][:19],
            info['pod_id'][:14] if info['pod_id'] else '-',
            status_colored,
            info.get('progress', '-')[:24],
            info['runtime'],
            info['start_time'],
            info['error'][:59] if info['error'] else '-'
        )
    
    # Create log panel with recent logs
    log_lines = list(log_queue)[-max_logs:]  # Get last N logs
    log_text = "\n".join(log_lines) if log_lines else "[dim]No logs yet...[/dim]"
    log_panel = Panel(
        log_text,
        title="📝 Activity Log",
        border_style="blue",
        padding=(1, 2)
    )

    # Create layout
    layout = Layout()
    layout.split_column(
        Layout(table, name="status", ratio=2),
        Layout(log_panel, name="logs", ratio=1)
    )

    return layout


def run_multiple_pods(pod_configs, API_KEY):
    """Run multiple pods in parallel with dashboard monitoring"""
    global _active_pods, _api_key

    manager = Manager()
    status_dict = manager.dict()
    log_queue = manager.list()  # Shared log queue
    _active_pods = manager.dict()  # Shared pod tracker for emergency cleanup
    _api_key = API_KEY  # Store API key globally for signal handlers
    processes = []
    
    console = Console()
    console.print(f"[cyan]Initializing {len(pod_configs)} pod(s)...[/cyan]")

    # Start all processes
    for idx, pod_config in enumerate(pod_configs):
        console.print(f"[yellow]Starting pod {idx}: {pod_config.get('name', 'unnamed')}[/yellow]")
        p = Process(target=run_single_pod, args=(pod_config, status_dict, idx, API_KEY, log_queue, _active_pods))
        p.start()
        processes.append(p)
        time.sleep(0.5)  # Stagger pod creation
    
    console.print("[green]All pods started. Monitoring...[/green]\n")

    # Monitor with live dashboard - faster refresh for smooth spinner
    interrupted = False
    try:
        with Live(generate_dashboard(status_dict, log_queue), refresh_per_second=4, console=console) as live:
            while any(p.is_alive() for p in processes):
                live.update(generate_dashboard(status_dict, log_queue))
                time.sleep(0.25)

            # Final update
            live.update(generate_dashboard(status_dict, log_queue))
    except KeyboardInterrupt:
        interrupted = True
        console.print("\n[red]⚠️  Interrupted by user (Ctrl+C)[/red]")
        console.print("[yellow]Cleaning up active pods...[/yellow]")

    # Wait for all processes to complete
    for p in processes:
        p.join(timeout=5)
        if p.is_alive():
            p.terminate()
    
    # Emergency cleanup if interrupted
    if interrupted:
        cleanup_all_pods()

    console.print("\n[green]✅ All pods completed![/green]\n")

    # Print final summary
    final_table = generate_dashboard(dict(status_dict), log_queue)
    console.print(final_table)

    # Generate summary statistics
    final_status = dict(status_dict)
    total = len(final_status)
    completed = sum(1 for s in final_status.values() if '✅ EXITED' in s['status'] or '🛑 TERMINATED' in s['status'])
    errors = sum(1 for s in final_status.values() if '❌ ERROR' in s['status'])
    timeouts = sum(1 for s in final_status.values() if '⏱️ TIMEOUT' in s['status'])

    console.print("\n" + "="*80)
    console.print("[bold cyan]📊 Summary Report[/bold cyan]")
    console.print("="*80)
    console.print(f"Total pods: [bold]{total}[/bold]")
    console.print(f"✅ Completed: [green]{completed}[/green]")
    console.print(f"❌ Errors: [red]{errors}[/red]")
    console.print(f"⏱️ Timeouts: [yellow]{timeouts}[/yellow]")
    console.print("="*80 + "\n")


def main(args):
    """Main entry point"""
    # Register cleanup handlers for emergency termination
    signal.signal(signal.SIGINT, cleanup_all_pods)   # Ctrl+C
    signal.signal(signal.SIGTERM, cleanup_all_pods)  # kill command
    atexit.register(cleanup_all_pods)  # Normal exit

    # .env 파일을 여러 경로에서 시도
    _env_candidates = [
        Path(__file__).parent / '.env',            # sweep/runpod/.env
        Path(__file__).parent.parent.parent / '.env',  # 프로젝트 루트/.env
        Path('/app/runpod/.env'),                  # RunPod 컨테이너 내부
    ]
    for _env_path in _env_candidates:
        if _env_path.exists():
            load_dotenv(_env_path)
            _console.print(f"[dim]Loaded .env from {_env_path}[/dim]")
            break

    API_KEY = os.getenv('RUNPOD_API_KEY')

    if not API_KEY:
        _console.print("[bold red]❌ RUNPOD_API_KEY not found![/bold red]")
        _console.print("[yellow]Searched .env locations:[/yellow]")
        for p in _env_candidates:
            exists = "✓" if p.exists() else "✗"
            _console.print(f"  {exists} {p}")
        _console.print("\n[yellow]Set it via: export RUNPOD_API_KEY=xxx 또는 .env 파일에 추가[/yellow]")
        return

    _console.print(f"[green]✅ API Key loaded (***{API_KEY[-4:]})[/green]")
    runpod.api_key = API_KEY

    pod_configs = []

    # -c/--config 는 이미 리스트로 들어온다
    config_args = args.config or []

    # 주어진 모든 config 에 대해 pod 설정 생성
    # 지원하는 형태:
    #   1) 폴더 이름  (e.g. "cpcgrl_reward_enum")  → config/<name>/*.yaml 로드
    #   2) 템플릿 이름 (e.g. "cpcgrl_reward_enum")  → template/<name>.yaml 전개 후 로드
    #   3) 단일 YAML 파일 경로
    for cfg in config_args:
        cfg_path = Path(cfg)

        # ── Case 1: config/ 아래 폴더가 이미 존재하면 그 안의 YAML 전부 로드
        config_dir = Path(config_folder_path) / cfg
        if config_dir.is_dir():
            _console.print(f"[cyan]📂 Loading configs from folder: {config_dir}[/cyan]")
            pod_configs.extend(_load_configs_from_dir(config_dir))
            continue

        # ── Case 2: template/ 아래 같은 이름의 템플릿이 있으면 전개 → 폴더 생성 → 로드
        template_path = Path(template_folder_path) / f"{cfg}.yaml"
        if not template_path.exists() and not cfg_path.suffix:
            # .yaml 없이 들어온 경우 config 폴더에서도 시도
            template_path = Path(template_folder_path) / cfg
        if template_path.exists() and template_path.is_file():
            _console.print(f"[cyan]📋 Expanding template: {template_path}[/cyan]")
            generated_dir = expand_and_save_sweep(template_path)
            pod_configs.extend(_load_configs_from_dir(generated_dir))
            continue

        # ── Case 3: 직접 YAML 파일 경로
        if not cfg_path.exists():
            cfg_path = Path(config_folder_path) / cfg
        if not cfg_path.exists():
            raise FileNotFoundError(
                f"Config not found: '{cfg}'\n"
                f"  Searched: config/{cfg}/, template/{cfg}.yaml, {cfg}"
            )

        with open(cfg_path, "r") as f:
            config_data = yaml.safe_load(f)
        pod_configs.append(create_pod_config_from_yaml(config_data))

    if pod_configs:
        print(f"Starting {len(pod_configs)} pod(s) with dashboard...")
        run_multiple_pods(pod_configs, API_KEY)
    else:
        print("No valid pod configurations found.")


if __name__ == '__main__':
    args = parse_args()
    try:
        main(args)
    except Exception as e:
        _console.print(f"\n[bold red]💥 Fatal error: {str(e)}[/bold red]")
        _console.print("[yellow]Attempting emergency cleanup...[/yellow]")
        cleanup_all_pods()
        raise
    finally:
        # Final safety check - cleanup any remaining pods
        if _active_pods:
            _console.print("[yellow]⚠️  Cleaning up remaining pods...[/yellow]")
            cleanup_all_pods()

"""
Usage:

Sweep execution (template auto-expand → config folder → parallel pods):
python deploy.py -c cpcgrl_reward_enum

  1) template/cpcgrl_reward_enum.yaml 의 sweep 섹션을 읽어
  2) config/cpcgrl_reward_enum/ 폴더에 개별 YAML 생성
  3) 폴더 안 모든 YAML을 병렬 pod으로 실행

이미 config 폴더가 있으면 템플릿 전개 없이 바로 실행:
python deploy.py -c cpcgrl_reward_enum

Single config execution:
python deploy.py -c train_ep20_bs512.yaml

Multiple configs:
python deploy.py -c config1.yaml config2.yaml config3.yaml
"""