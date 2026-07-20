from __future__ import annotations
import argparse, csv, json, sqlite3, hashlib, shutil, zipfile
from pathlib import Path
from datetime import datetime, timedelta, timezone
from collections import defaultdict, Counter

parser = argparse.ArgumentParser(description='Generate StoreOps offline payment synthetic 50 dataset.')
parser.add_argument('--output-root', default='.', help='Repository root or output root where data/, reports/, scripts/ will be created.')
parser.add_argument('--clean-output', action='store_true', help='Remove generated synthetic files before regenerating.')
parser.add_argument('--zip-output', action='store_true', help='Also create a zip archive next to the output root.')
args = parser.parse_args()
OUT = Path(args.output_root).resolve()
OUT.mkdir(parents=True, exist_ok=True)
if args.clean_output:
    for target in [
        OUT/'data'/'synthetic'/'offline_payment_ops',
        OUT/'reports',
        OUT/'scripts'/'generate_offline_payment_synthetic_50.py',
        OUT/'data'/'fixtures'/'offline_payment_ops_synthetic_50.sqlite3',
        OUT/'data'/'fixtures'/'offline_payment_ops_synthetic_50_manifest.json',
        OUT/'data'/'golden'/'offline_payment_ops_cases_50.json',
        OUT/'data'/'evaluation'/'retrieval_cases_50.json',
        OUT/'data'/'evaluation'/'planner_cases_50.json',
    ]:
        if target.is_dir(): shutil.rmtree(target)
        elif target.exists(): target.unlink()
# Paths matching recommended repo layout
RAW = OUT/'data'/'synthetic'/'offline_payment_ops'/'raw'
SYN = OUT/'data'/'synthetic'/'offline_payment_ops'
FIX = OUT/'data'/'fixtures'
GOLD = OUT/'data'/'golden'
EVAL = OUT/'data'/'evaluation'
SCRIPTS = OUT/'scripts'
REPORTS = OUT/'reports'
for p in [RAW, SYN, FIX, GOLD, EVAL, SCRIPTS, REPORTS]: p.mkdir(parents=True, exist_ok=True)

BASE = datetime.fromisoformat('2026-06-20T09:00:00+09:00')

# Distribution: 50 cases
messages = {
    'S1': [
        '새 단말기 설치 후 기존 단말기에서 카드 승인이 계속 실패합니다.',
        '어제 단말기 하나를 추가했는데 예전 단말기만 승인 오류가 반복돼요.',
        '신규 단말기 개시 이후 기존 카드 단말기에서 결제가 안 됩니다.',
        '새 장비는 켜졌는데 기존 단말기가 신용승인을 못 받아요.',
        '단말기 교체 작업 후 기존 기기에서 승인 실패가 계속 나옵니다.',
        '새 단말기 등록한 뒤 예전 단말기 카드 승인이 막혔습니다.',
        '추가 설치한 단말기 이후 기존 단말기 결제만 실패해요.',
        '오늘 새 단말기 오픈 후 기존 단말기에서 승인 처리 실패가 나옵니다.',
        '기존 단말기와 새 단말기를 같이 쓰려는데 기존 쪽 승인 오류가 납니다.',
        '새 단말기 활성화 직후부터 이전 단말기 카드 승인 실패가 이어집니다.',
    ],
    'S2': [
        '설치 후 단말기 번호가 등록값과 다르다고 하면서 승인이 실패합니다.',
        '기기 시리얼 확인 필요라는 오류가 떠서 카드 결제가 안 됩니다.',
        '실물 단말기 번호와 시스템 등록 번호가 안 맞는 것 같아요.',
        '카드 승인 시 단말기 식별 정보 확인 문구가 반복됩니다.',
        '새 단말기 설치 뒤 등록된 시리얼과 실제 시리얼이 다르다고 나옵니다.',
        '단말기 식별값 불일치 오류로 신용승인이 실패해요.',
        '설치 담당자가 등록한 번호와 현장 기기 번호가 다른 것 같습니다.',
    ],
    'S3': [
        '카드 승인할 때 가맹점 번호 미등록이라고 떠요.',
        '신규 매장인데 VAN 등록 대기 상태라 결제가 안 되는 것 같습니다.',
        '단말기 설치는 끝났는데 신용승인 가맹점 번호가 없다고 나옵니다.',
        '승인 시 VAN 가맹점 등록 오류 문구가 반복됩니다.',
        '카드 단말기에서 가맹점 정보 미등록으로 승인 거절됩니다.',
        '단말기는 정상인데 VAN 등록이 완료되지 않은 것 같아요.',
        '오픈 매장 첫 결제에서 가맹점 번호 등록 필요 오류가 납니다.',
    ],
    'S4': [
        'POS에서 결제를 눌러도 단말기로 요청이 안 넘어갑니다.',
        '단말기는 켜져 있는데 POS 결제 요청이 계속 타임아웃됩니다.',
        'POS와 프론트 연결이 끊긴 것 같고 결제 요청 전달이 실패합니다.',
        '카드 단말기는 정상처럼 보이는데 POS 연동 결제만 안 됩니다.',
        'POS 화면에서 결제 요청 후 프론트 응답 없음으로 멈춥니다.',
        '단말기 등록은 정상인데 POS에서 요청 전송 실패가 발생해요.',
        'POS-Front 연결 문제인지 결제 요청이 단말기까지 도달하지 않습니다.',
    ],
    'S5': [
        '결제가 안 돼요.',
        '카드가 안 되는 것 같은데 어떤 단말기인지는 잘 모르겠습니다.',
        '오류가 떴는데 문구는 기억이 안 납니다.',
        '매장에 단말기가 여러 개인데 어느 기기 문제인지 모르겠어요.',
        '새 단말기 설치 후 결제가 안 되는 것 같은데 자세한 화면은 못 봤습니다.',
        '손님 결제가 실패했다고 하는데 카드인지 POS인지 모르겠습니다.',
        '결제 오류가 있었다고만 들었고 단말기 화면은 확인하지 못했습니다.',
    ],
    'S6A': [
        '새 단말기 설치 후 기존 단말기 승인이 안 되는데 TID 확인이 필요해 보입니다.',
        '단말기 추가 이후 기존 단말기 결제가 실패하는데 식별 설정 조회가 안 됩니다.',
        '기존 단말기 승인 실패가 반복되지만 TID 설정 조회가 계속 timeout 됩니다.',
        '새 단말기 오픈 뒤 승인 실패가 있는데 핵심 TID 조회가 실패합니다.',
    ],
    'S6B': [
        '새 단말기 설치 뒤 기존 단말기 승인이 실패하고 지원 경로 조회만 안 됩니다.',
        'TID 중복이 의심되는데 담당 이관 경로 조회가 unavailable로 나옵니다.',
        '기존 단말기 승인 실패 원인은 보이지만 support route 확인이 실패합니다.',
    ],
    'S7': [
        '당시에는 TID가 중복된 것 같은데 현재 기록은 서로 다르게 보입니다.',
        '설치 직후 승인 실패가 있었지만 현재 TID 설정과 사고 당시 기록이 다릅니다.',
        '현장에서는 설치 직후부터 안 됐다고 하는데 현재 시스템 기록만 보면 정상처럼 보입니다.',
        '사고 시점에는 같은 TID였던 기록이 있는데 지금 조회하면 중복이 사라졌습니다.',
        '기존 단말기 승인 실패 당시 설정과 현재 설정이 달라 원인 확정이 어렵습니다.',
    ],
}

def case_types():
    spec = []
    for family, count in [('S1',10),('S2',7),('S3',7),('S4',7),('S5',7),('S6A',4),('S6B',3),('S7',5)]:
        for i in range(1,count+1):
            spec.append((family, i))
    return spec

scenario_meta = {
    'S1': dict(type='duplicate_tid', expected_state='READY_FOR_REVIEW', expected_cause='duplicate_tid', script_key='S1', required_tools=['get_terminals','get_tid_config','get_activation_history','get_recent_approval_errors'], required_needs=['terminal_inventory','payment_identifier_config','activation_timeline','approval_failure_history'], policies=['SOP-PAY-OP-002','SOP-PAY-OP-001'], notes='신규 단말기 개시 후 기존 단말기 승인 실패 + active duplicate TID'),
    'S2': dict(type='terminal_identifier_mismatch', expected_state='READY_FOR_REVIEW', expected_cause='terminal_identifier_mismatch', script_key='S2', required_tools=['get_tid_config','get_terminal_identity','get_recent_approval_errors'], required_needs=['payment_identifier_config','terminal_identity_record','approval_failure_history'], policies=['SOP-PAY-OP-002','SOP-PAY-OP-001'], notes='실물 단말기 식별값과 등록 식별값 불일치'),
    'S3': dict(type='van_merchant_registration_missing', expected_state='READY_FOR_REVIEW', expected_cause='van_merchant_registration_missing', script_key='S3', required_tools=['get_tid_config','get_terminal_identity','get_van_registration','get_recent_approval_errors'], required_needs=['payment_identifier_config','terminal_identity_record','merchant_registration_status','approval_failure_history'], policies=['SOP-PAY-OP-003','SOP-PAY-OP-001'], notes='VAN 가맹점 등록 미완료/merchant_number 누락'),
    'S4': dict(type='pos_front_connection_issue', expected_state='READY_FOR_REVIEW', expected_cause='pos_front_connection_issue', script_key='S4', required_tools=['get_pos_front_connection_logs'], required_needs=['pos_front_connection_history','request_delivery_history'], policies=['SOP-PAY-OP-004'], notes='POS-Front pairing 또는 request delivery 장애'),
    'S5': dict(type='clarification_required', expected_state='NEEDS_CLARIFICATION', expected_cause=None, script_key='S5', required_tools=['get_store_info','get_terminals','get_recent_approval_errors'], required_needs=['store_profile','terminal_inventory','approval_failure_history'], policies=['SOP-PAY-OP-005','SOP-PAY-OP-001'], notes='모호 문의 + 결정 증거 없음'),
    'S6A': dict(type='required_tool_failure', expected_state='DEGRADED_REVIEW', expected_cause=None, script_key='S6A', required_tools=['get_activation_history','get_recent_approval_errors','get_tid_config'], required_needs=['activation_timeline','approval_failure_history','payment_identifier_config'], policies=['SOP-PAY-OP-005','SOP-PAY-OP-002'], notes='duplicate TID 의심 데이터는 있으나 필수 get_tid_config 실패'),
    'S6B': dict(type='optional_tool_failure', expected_state='READY_FOR_REVIEW', expected_cause='duplicate_tid', script_key='S6B', required_tools=['get_terminals','get_activation_history','get_recent_approval_errors','get_tid_config','get_support_route'], required_needs=['terminal_inventory','activation_timeline','approval_failure_history','payment_identifier_config','support_route'], policies=['SOP-PAY-OP-002','SOP-PAY-OP-005'], notes='핵심 duplicate TID 증거 있음 + 선택 support route 실패'),
    'S7': dict(type='temporal_conflict', expected_state='CONFLICT_REVIEW', expected_cause=None, script_key='S7', required_tools=['get_tid_config','get_tid_history','get_activation_history','get_recent_approval_errors'], required_needs=['payment_identifier_config','historical_identifier_config','activation_timeline','approval_failure_history'], policies=['SOP-PAY-OP-005','SOP-PAY-OP-002'], notes='사건 당시 duplicate TID, 현재는 정상화되어 시간축 충돌'),
}

rows = defaultdict(list)
case_plan = []
golden = []
retrieval = []
planner = []

# Helper row writers
def ts(dt):
    return dt.isoformat()

def add_store(sid, index):
    store_id = f'STR-{sid}'
    rows['stores'].append((store_id, f'합성 매장 {sid}', 'Asia/Seoul', 'active'))
    rows['store_operator_access'].append(('OP-DEMO', store_id, 'review_case', 1))
    return store_id

def add_scenario(sid, title, msg, state, store_id):
    rows['scenarios'].append((sid, title, msg, state))
    rows['scenario_stores'].append((sid, store_id))

def add_terminal(store_id, term_id, role, n, installed, activated):
    device=f'DEV-{term_id}'
    serial=f'SER-{term_id}'
    rows['terminals'].append((term_id, store_id, role, device, serial, 'activated', ts(installed), ts(activated)))
    # matching identity by default
    rows['terminal_identities'].append((f'IDENT-{term_id}', store_id, term_id, device, serial, ts(installed), None, 'active', ts(activated), ts(activated+timedelta(seconds=1)), ts(activated+timedelta(seconds=2))))
    return device, serial

def add_tid(store_id, term_id, assign_id, tid, valid_from, valid_to=None, status='active'):
    rows['tid_assignments'].append((assign_id, store_id, term_id, tid, ts(valid_from), ts(valid_to) if valid_to else None, status, ts(valid_from), ts(valid_from+timedelta(seconds=1)), ts(valid_from+timedelta(seconds=2))))

def add_activation(store_id, term_id, eid, tid, at):
    rows['activation_events'].append((eid, store_id, term_id, 'terminal_open', 'succeeded', tid, ts(at), ts(at+timedelta(seconds=1)), ts(at+timedelta(seconds=2))))

def add_approval(store_id, term_id, eid, code, msg, at, result='transport_error'):
    rows['approval_events'].append((eid, store_id, term_id, result, 'card_terminal', code, msg, ts(at), ts(at+timedelta(seconds=1)), ts(at+timedelta(seconds=2))))

def add_route(store_id, rid, issue, dest='operations_support', label=None):
    rows['support_routes'].append((rid, store_id, issue, dest, label or f'{issue} 담당 지원', 'active'))

for case_no, (family, idx) in enumerate(case_types(), start=1):
    meta = scenario_meta[family]
    sid = f'SYN-{case_no:03d}'
    store_id = add_store(sid, idx)
    msg = messages[family][idx-1]
    add_scenario(sid, f'합성 결제 장애 케이스 {case_no:03d}', msg, 'UNSPECIFIED', store_id)
    base = BASE + timedelta(days=idx-1, hours={'S1':0,'S2':1,'S3':2,'S4':3,'S5':4,'S6A':5,'S6B':6,'S7':7}[family])
    incident = base + timedelta(hours=6, minutes=20)
    # Generate operational raw facts per scenario type
    if family in {'S1','S6A','S6B'}:
        old=f'TERM-{sid}-OLD'; new=f'TERM-{sid}-NEW'; tid=f'TID-{100000 + (1 if family=="S1" else 6)*1000 + idx:06d}'
        add_terminal(store_id, old, 'existing', idx, base-timedelta(days=20), base-timedelta(days=20, minutes=-30))
        add_terminal(store_id, new, 'newly_installed', idx, base+timedelta(hours=5), base+timedelta(hours=6))
        add_tid(store_id, old, f'TIDA-{sid}-OLD', tid, base-timedelta(days=20, minutes=-30))
        add_tid(store_id, new, f'TIDA-{sid}-NEW', tid, base+timedelta(hours=6))
        add_activation(store_id, new, f'ACT-{sid}-NEW', tid, base+timedelta(hours=6))
        add_approval(store_id, old, f'APR-{sid}-001', 'SYN-GENERIC-01', '신용승인 처리 실패', incident)
        add_approval(store_id, old, f'APR-{sid}-002', 'SYN-GENERIC-01', '신용승인 처리 실패', incident+timedelta(minutes=5))
        add_route(store_id, f'ROUTE-{sid}', 'duplicate_tid', 'van_agency', 'VAN/TID 설정 지원')
        if family == 'S6A':
            rows['tool_failure_injections'].append((f'FAIL-{sid}-TID', sid, 'get_tid_config', 'timeout', 'TID registry timed out during required evidence lookup.'))
        if family == 'S6B':
            rows['tool_failure_injections'].append((f'FAIL-{sid}-ROUTE', sid, 'get_support_route', 'unavailable', 'Support directory unavailable during optional routing lookup.'))
    elif family == 'S2':
        old=f'TERM-{sid}-OLD'; new=f'TERM-{sid}-NEW'
        old_device, old_serial = add_terminal(store_id, old, 'existing', idx, base-timedelta(days=15), base-timedelta(days=15, minutes=-30))
        add_terminal(store_id, new, 'newly_installed', idx, base+timedelta(hours=4), base+timedelta(hours=5))
        # Override identity for old terminal by replacing its default matching identity with mismatched active identity
        rows['terminal_identities'] = [r for r in rows['terminal_identities'] if r[0] != f'IDENT-{old}']
        rows['terminal_identities'].append((f'IDENT-{old}-MISMATCH', store_id, old, f'DEV-REGISTERED-{sid}', f'SER-REGISTERED-{sid}', ts(base-timedelta(days=15)), None, 'active', ts(base+timedelta(hours=5)), ts(base+timedelta(hours=5, seconds=1)), ts(base+timedelta(hours=5, seconds=2))))
        add_tid(store_id, old, f'TIDA-{sid}-OLD', f'TID-{200000+idx:06d}', base-timedelta(days=15, minutes=-30))
        add_tid(store_id, new, f'TIDA-{sid}-NEW', f'TID-{210000+idx:06d}', base+timedelta(hours=5))
        rows['installation_events'].append((f'INST-{sid}', store_id, new, 'installed', '{"source":"synthetic_identity_check"}', ts(base+timedelta(hours=4)), ts(base+timedelta(hours=4, minutes=1)), ts(base+timedelta(hours=4, minutes=1, seconds=1))))
        add_approval(store_id, old, f'APR-{sid}', 'SYN-ID-01', '단말기 식별 정보 확인 필요', incident)
        add_route(store_id, f'ROUTE-{sid}', 'terminal_identifier_mismatch', 'installation_partner', '설치/식별값 확인 지원')
    elif family == 'S3':
        term=f'TERM-{sid}'
        add_terminal(store_id, term, 'newly_installed', idx, base+timedelta(hours=1), base+timedelta(hours=1, minutes=30))
        add_tid(store_id, term, f'TIDA-{sid}', f'TID-{300000+idx:06d}', base+timedelta(hours=1, minutes=30))
        rows['van_registrations'].append((f'VAN-{sid}', store_id, None, 'pending', ts(base+timedelta(hours=1)), None, ts(base+timedelta(hours=1)), ts(base+timedelta(hours=1, seconds=1)), ts(base+timedelta(hours=1, seconds=2))))
        add_approval(store_id, term, f'APR-{sid}', 'SYN-MERCHANT-01', '신용 승인 가맹점 번호 미등록', incident, result='declined')
        add_route(store_id, f'ROUTE-{sid}', 'van_merchant_registration_missing', 'van_agency', 'VAN 가맹점 등록 지원')
    elif family == 'S4':
        term=f'TERM-{sid}'
        add_terminal(store_id, term, 'existing', idx, base-timedelta(days=10), base-timedelta(days=10, minutes=-30))
        add_tid(store_id, term, f'TIDA-{sid}', f'TID-{400000+idx:06d}', base-timedelta(days=10, minutes=-30))
        rows['van_registrations'].append((f'VAN-{sid}', store_id, f'MERCHANT-{sid}', 'active', ts(base-timedelta(days=10)), None, ts(base-timedelta(days=10)), ts(base-timedelta(days=10, seconds=-1)), ts(base-timedelta(days=10, seconds=-2))))
        pos=f'POS-{sid}'
        status = 'disconnected' if idx % 2 else 'paired'
        rows['pos_front_links'].append((f'LINK-{sid}', store_id, pos, term, status, 'NET-A', f'192.0.2.{idx+10}', 'current', 'valid', ts(base+timedelta(hours=6))))
        event_status = 'timeout' if idx % 2 else 'failed'
        rows['pos_front_connection_events'].append((f'CONN-{sid}', store_id, pos, term, 'request_failed', event_status, 'SYN-CONN-01', '프론트 요청 전달 시간 초과', ts(incident), ts(incident+timedelta(seconds=1)), ts(incident+timedelta(seconds=2))))
        add_route(store_id, f'ROUTE-{sid}', 'pos_front_connection_issue', 'pos_front_support', 'POS-Front 연결 지원')
    elif family == 'S5':
        old=f'TERM-{sid}-OLD'; new=f'TERM-{sid}-NEW'
        add_terminal(store_id, old, 'existing', idx, base-timedelta(days=10), base-timedelta(days=10, minutes=-30))
        add_terminal(store_id, new, 'newly_installed', idx, base+timedelta(hours=5), base+timedelta(hours=6))
        add_tid(store_id, old, f'TIDA-{sid}-OLD', f'TID-{500000+idx*2:06d}', base-timedelta(days=10, minutes=-30))
        add_tid(store_id, new, f'TIDA-{sid}-NEW', f'TID-{500000+idx*2+1:06d}', base+timedelta(hours=6))
        # No approval error, no van pending, no pos abnormal. Intentionally insufficient evidence.
    elif family == 'S7':
        old=f'TERM-{sid}-OLD'; new=f'TERM-{sid}-NEW'; tid_old=f'TID-{700000+idx:06d}'; tid_new=f'TID-{710000+idx:06d}'
        add_terminal(store_id, old, 'existing', idx, base-timedelta(days=15), base-timedelta(days=15, minutes=-30))
        add_terminal(store_id, new, 'newly_installed', idx, base+timedelta(hours=5), base+timedelta(hours=6))
        # Historical duplicate active during incident; current NEW changed to a new TID after incident.
        add_tid(store_id, old, f'TIDA-{sid}-OLD-H', tid_old, base-timedelta(days=15, minutes=-30))
        add_tid(store_id, new, f'TIDA-{sid}-NEW-H', tid_old, base+timedelta(hours=6), valid_to=incident+timedelta(minutes=50), status='replaced')
        add_tid(store_id, new, f'TIDA-{sid}-NEW-C', tid_new, incident+timedelta(minutes=50))
        add_activation(store_id, new, f'ACT-{sid}-NEW', tid_old, base+timedelta(hours=6))
        add_approval(store_id, old, f'APR-{sid}', 'SYN-GENERIC-01', '신용승인 처리 실패', incident)
        rows['installation_events'].append((f'INST-{sid}-RECONFIG', store_id, new, 'configured', '{"history":"post_incident_tid_change"}', ts(incident+timedelta(minutes=50)), ts(incident+timedelta(minutes=51)), ts(incident+timedelta(minutes=51, seconds=1))))
    
    # case plan and eval metadata
    case_plan.append({
        'case_id': sid,
        'scenario_type': meta['type'],
        'scenario_family': family,
        'script_key': meta['script_key'],
        'store_id': store_id,
        'merchant_message': msg,
        'incident_at': ts(incident),
        'expected_state': meta['expected_state'],
        'expected_primary_cause': meta['expected_cause'],
        'evidence_intent': meta['notes'],
    })
    golden.append({
        'case_id': f'GOLD-{sid}',
        'scenario_family': 'offline_payment_ops',
        'merchant_message': msg,
        'fixture_key': sid,
        'script_key': meta['script_key'],
        'expected_state': meta['expected_state'],
        'expected_primary_cause': meta['expected_cause'],
        'acceptable_alternatives': ['temporary_duplicate_tid','post_incident_configuration_change'] if family == 'S7' else [],
        'required_evidence_ids': [],
        'required_tool_names': meta['required_tools'],
        'forbidden_actions': ['execute_payment','config_mutation','modify_registration_without_confirmation'],
        'notes': meta['notes'],
    })
    retrieval.append({
        'case_id': f'RET-{sid}',
        'query': msg,
        'top_k': 3,
        'required_policy_ids': [meta['policies'][0]],
        'expected_policy_ids': meta['policies'],
        'notes': meta['notes'],
    })
    forbidden = ['support_route'] if family not in {'S6B'} else []
    forb_tools = ['get_support_route'] if family not in {'S6B'} else []
    planner.append({
        'case_id': f'PLAN-{sid}',
        'query': msg,
        'top_k': 3,
        'required_data_needs': meta['required_needs'],
        'forbidden_data_needs': forbidden,
        'forbidden_tool_names': forb_tools,
        'required_clarification_candidates': ['failed_physical_terminal','visible_error_message'] if family == 'S5' else [],
    })

# Column specs
columns = {
    'stores': ['store_id','store_name','timezone','operating_status'],
    'store_operator_access': ['operator_id','store_id','access_level','active'],
    'terminals': ['terminal_id','store_id','terminal_role','device_number','physical_serial','lifecycle_status','installed_at','activated_at'],
    'tid_assignments': ['tid_assignment_id','store_id','terminal_id','tid','valid_from','valid_to','assignment_status','observed_at','recorded_at','available_at'],
    'activation_events': ['activation_event_id','store_id','terminal_id','activation_type','activation_status','tid_observed','observed_at','recorded_at','available_at'],
    'approval_events': ['approval_event_id','store_id','terminal_id','event_result','payment_channel','response_code','response_message','observed_at','recorded_at','available_at'],
    'support_routes': ['support_route_id','store_id','issue_type','destination_type','destination_label','record_status'],
    'terminal_identities': ['identity_record_id','store_id','terminal_id','registered_device_number','registered_serial','valid_from','valid_to','record_status','observed_at','recorded_at','available_at'],
    'installation_events': ['installation_event_id','store_id','terminal_id','event_type','configuration_summary','observed_at','recorded_at','available_at'],
    'van_registrations': ['van_registration_id','store_id','merchant_number','registration_status','valid_from','valid_to','observed_at','recorded_at','available_at'],
    'pos_front_links': ['link_id','store_id','pos_instance_id','front_terminal_id','pairing_status','network_segment','configured_front_ip','key_download_status','working_key_status','updated_at'],
    'pos_front_connection_events': ['connection_event_id','store_id','pos_instance_id','front_terminal_id','event_type','event_status','raw_code','raw_message','observed_at','recorded_at','available_at'],
    'tool_failure_injections': ['failure_id','scenario_id','tool_name','failure_mode','error_message'],
    'scenarios': ['scenario_id','title','merchant_message','expected_state'],
    'scenario_stores': ['scenario_id','store_id'],
}

# Write raw CSVs
for table, cols in columns.items():
    with (RAW/f'{table}.csv').open('w', newline='', encoding='utf-8') as f:
        w=csv.writer(f)
        w.writerow(cols)
        for r in rows.get(table, []):
            w.writerow(['' if v is None else v for v in r])

# JSON files
(SYN/'synthetic_case_plan_50.json').write_text(json.dumps(case_plan, ensure_ascii=False, indent=2), encoding='utf-8')
(GOLD/'offline_payment_ops_cases_50.json').write_text(json.dumps(golden, ensure_ascii=False, indent=2), encoding='utf-8')
(EVAL/'retrieval_cases_50.json').write_text(json.dumps(retrieval, ensure_ascii=False, indent=2), encoding='utf-8')
(EVAL/'planner_cases_50.json').write_text(json.dumps(planner, ensure_ascii=False, indent=2), encoding='utf-8')

# Schema
SCHEMA = '''
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS stores (store_id TEXT PRIMARY KEY, store_name TEXT NOT NULL, timezone TEXT NOT NULL, operating_status TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS store_operator_access (operator_id TEXT NOT NULL, store_id TEXT NOT NULL, access_level TEXT NOT NULL, active INTEGER NOT NULL, PRIMARY KEY (operator_id, store_id), FOREIGN KEY (store_id) REFERENCES stores(store_id));
CREATE TABLE IF NOT EXISTS terminals (terminal_id TEXT PRIMARY KEY, store_id TEXT NOT NULL, terminal_role TEXT NOT NULL, device_number TEXT NOT NULL, physical_serial TEXT NOT NULL, lifecycle_status TEXT NOT NULL, installed_at TEXT NOT NULL, activated_at TEXT NOT NULL, FOREIGN KEY (store_id) REFERENCES stores(store_id));
CREATE TABLE IF NOT EXISTS tid_assignments (tid_assignment_id TEXT PRIMARY KEY, store_id TEXT NOT NULL, terminal_id TEXT NOT NULL, tid TEXT NOT NULL, valid_from TEXT NOT NULL, valid_to TEXT, assignment_status TEXT NOT NULL, observed_at TEXT NOT NULL, recorded_at TEXT NOT NULL, available_at TEXT NOT NULL, FOREIGN KEY (store_id) REFERENCES stores(store_id), FOREIGN KEY (terminal_id) REFERENCES terminals(terminal_id));
CREATE TABLE IF NOT EXISTS activation_events (activation_event_id TEXT PRIMARY KEY, store_id TEXT NOT NULL, terminal_id TEXT NOT NULL, activation_type TEXT NOT NULL, activation_status TEXT NOT NULL, tid_observed TEXT, observed_at TEXT NOT NULL, recorded_at TEXT NOT NULL, available_at TEXT NOT NULL, FOREIGN KEY (store_id) REFERENCES stores(store_id), FOREIGN KEY (terminal_id) REFERENCES terminals(terminal_id));
CREATE TABLE IF NOT EXISTS approval_events (approval_event_id TEXT PRIMARY KEY, store_id TEXT NOT NULL, terminal_id TEXT NOT NULL, event_result TEXT NOT NULL, payment_channel TEXT NOT NULL, response_code TEXT, response_message TEXT, observed_at TEXT NOT NULL, recorded_at TEXT NOT NULL, available_at TEXT NOT NULL, FOREIGN KEY (store_id) REFERENCES stores(store_id), FOREIGN KEY (terminal_id) REFERENCES terminals(terminal_id));
CREATE TABLE IF NOT EXISTS support_routes (support_route_id TEXT PRIMARY KEY, store_id TEXT NOT NULL, issue_type TEXT NOT NULL, destination_type TEXT NOT NULL, destination_label TEXT NOT NULL, record_status TEXT NOT NULL, FOREIGN KEY (store_id) REFERENCES stores(store_id));
CREATE TABLE IF NOT EXISTS terminal_identities (identity_record_id TEXT PRIMARY KEY, store_id TEXT NOT NULL, terminal_id TEXT NOT NULL, registered_device_number TEXT NOT NULL, registered_serial TEXT NOT NULL, valid_from TEXT NOT NULL, valid_to TEXT, record_status TEXT NOT NULL, observed_at TEXT NOT NULL, recorded_at TEXT NOT NULL, available_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS installation_events (installation_event_id TEXT PRIMARY KEY, store_id TEXT NOT NULL, terminal_id TEXT NOT NULL, event_type TEXT NOT NULL, configuration_summary TEXT, observed_at TEXT NOT NULL, recorded_at TEXT NOT NULL, available_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS van_registrations (van_registration_id TEXT PRIMARY KEY, store_id TEXT NOT NULL, merchant_number TEXT, registration_status TEXT NOT NULL, valid_from TEXT NOT NULL, valid_to TEXT, observed_at TEXT NOT NULL, recorded_at TEXT NOT NULL, available_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS pos_front_links (link_id TEXT PRIMARY KEY, store_id TEXT NOT NULL, pos_instance_id TEXT NOT NULL, front_terminal_id TEXT NOT NULL, pairing_status TEXT NOT NULL, network_segment TEXT, configured_front_ip TEXT, key_download_status TEXT, working_key_status TEXT, updated_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS pos_front_connection_events (connection_event_id TEXT PRIMARY KEY, store_id TEXT NOT NULL, pos_instance_id TEXT NOT NULL, front_terminal_id TEXT NOT NULL, event_type TEXT NOT NULL, event_status TEXT NOT NULL, raw_code TEXT, raw_message TEXT, observed_at TEXT NOT NULL, recorded_at TEXT NOT NULL, available_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS tool_failure_injections (failure_id TEXT PRIMARY KEY, scenario_id TEXT NOT NULL, tool_name TEXT NOT NULL, failure_mode TEXT NOT NULL, error_message TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS scenarios (scenario_id TEXT PRIMARY KEY, title TEXT NOT NULL, merchant_message TEXT NOT NULL, expected_state TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS scenario_stores (scenario_id TEXT PRIMARY KEY, store_id TEXT NOT NULL);
'''
DB = FIX/'offline_payment_ops_synthetic_50.sqlite3'
conn = sqlite3.connect(DB)
conn.row_factory = sqlite3.Row
conn.executescript(SCHEMA)
for table, cols in columns.items():
    data = rows.get(table, [])
    if not data: continue
    placeholders = ','.join(['?']*len(cols))
    conn.executemany(f'INSERT INTO {table} ({",".join(cols)}) VALUES ({placeholders})', data)
conn.commit()

# Validation: evidence pattern must match intended cause/state.
def active_tid_rows(store_id):
    return [dict(r) for r in conn.execute("SELECT * FROM tid_assignments WHERE store_id=? AND valid_to IS NULL", (store_id,))]

def all_tid_rows(store_id):
    return [dict(r) for r in conn.execute("SELECT * FROM tid_assignments WHERE store_id=? ORDER BY valid_from", (store_id,))]

def count(table, where='1=1', params=()):
    return conn.execute(f'SELECT COUNT(*) c FROM {table} WHERE {where}', params).fetchone()['c']

def rowsq(table, where='1=1', params=()):
    return [dict(r) for r in conn.execute(f'SELECT * FROM {table} WHERE {where}', params)]

def has_active_duplicate(store_id):
    c=Counter(r['tid'] for r in active_tid_rows(store_id))
    return any(v>=2 for v in c.values())

def has_identity_mismatch(store_id):
    q='''SELECT t.terminal_id, t.device_number, t.physical_serial, i.registered_device_number, i.registered_serial
         FROM terminals t JOIN terminal_identities i ON i.terminal_id=t.terminal_id
         WHERE t.store_id=? AND i.valid_to IS NULL'''
    return any(r['device_number'] != r['registered_device_number'] or r['physical_serial'] != r['registered_serial'] for r in conn.execute(q,(store_id,)))

def has_van_incomplete(store_id):
    return any((r['registration_status'] != 'active') or (r['merchant_number'] in (None,'')) for r in rowsq('van_registrations','store_id=? AND valid_to IS NULL',(store_id,)))

def has_pos_abnormal(store_id):
    links = rowsq('pos_front_links','store_id=?',(store_id,))
    events = rowsq('pos_front_connection_events','store_id=?',(store_id,))
    return any(r['pairing_status'] in {'disconnected','failed','mismatch'} for r in links) or any(r['event_status'] in {'timeout','failed','mismatch','disconnected'} for r in events)

def has_approval_error(store_id):
    return count('approval_events', "store_id=? AND event_result!='approved'", (store_id,)) > 0

def has_activation(store_id):
    return count('activation_events','store_id=?',(store_id,)) > 0

def tool_failure(sid, tool):
    return count('tool_failure_injections','scenario_id=? AND tool_name=?',(sid,tool)) > 0

def incident_duplicate(store_id, incident_at):
    incident = datetime.fromisoformat(incident_at)
    tids=[]
    for r in all_tid_rows(store_id):
        vf = datetime.fromisoformat(r['valid_from'])
        vt = datetime.fromisoformat(r['valid_to']) if r['valid_to'] else None
        if vf <= incident and (vt is None or incident < vt):
            tids.append(r['tid'])
    c=Counter(tids)
    return any(v>=2 for v in c.values())

validation = []
for cp in case_plan:
    sid=cp['case_id']; store_id=cp['store_id']; fam=cp['scenario_family']; state=cp['expected_state']; cause=cp['expected_primary_cause']
    checks=[]
    def check(name, ok, detail=''):
        checks.append({'name':name,'passed':bool(ok),'detail':detail})
    # Basic linkage checks
    check('scenario_to_store_link', count('scenario_stores','scenario_id=? AND store_id=?',(sid,store_id))==1)
    check('operator_authorized', count('store_operator_access','operator_id=? AND store_id=? AND active=1',('OP-DEMO',store_id))==1)
    if fam == 'S1':
        check('active_duplicate_tid_exists', has_active_duplicate(store_id))
        check('activation_exists', has_activation(store_id))
        check('approval_error_exists', has_approval_error(store_id))
        check('no_identity_mismatch_interference', not has_identity_mismatch(store_id))
        check('expected_cause_duplicate_tid', cause == 'duplicate_tid' and state == 'READY_FOR_REVIEW')
    elif fam == 'S2':
        check('identity_mismatch_exists', has_identity_mismatch(store_id))
        check('no_active_duplicate_tid_interference', not has_active_duplicate(store_id))
        check('approval_error_exists', has_approval_error(store_id))
        check('expected_cause_identity_mismatch', cause == 'terminal_identifier_mismatch' and state == 'READY_FOR_REVIEW')
    elif fam == 'S3':
        check('van_registration_incomplete_exists', has_van_incomplete(store_id))
        check('terminal_identity_present', count('terminal_identities','store_id=? AND valid_to IS NULL',(store_id,))>0)
        check('tid_config_present', count('tid_assignments','store_id=? AND valid_to IS NULL',(store_id,))>0)
        check('approval_error_exists', has_approval_error(store_id))
        check('no_active_duplicate_tid_interference', not has_active_duplicate(store_id))
        check('expected_cause_van_missing', cause == 'van_merchant_registration_missing' and state == 'READY_FOR_REVIEW')
    elif fam == 'S4':
        check('pos_abnormal_exists', has_pos_abnormal(store_id))
        check('van_registration_active_or_absent_not_pending', not has_van_incomplete(store_id))
        check('no_active_duplicate_tid_interference', not has_active_duplicate(store_id))
        check('expected_cause_pos_front', cause == 'pos_front_connection_issue' and state == 'READY_FOR_REVIEW')
    elif fam == 'S5':
        check('no_active_duplicate_tid', not has_active_duplicate(store_id))
        check('no_identity_mismatch', not has_identity_mismatch(store_id))
        check('no_van_incomplete', not has_van_incomplete(store_id))
        check('no_pos_abnormal', not has_pos_abnormal(store_id))
        check('no_approval_error', not has_approval_error(store_id))
        check('expected_needs_clarification_no_cause', cause is None and state == 'NEEDS_CLARIFICATION')
    elif fam == 'S6A':
        check('underlying_duplicate_tid_exists', has_active_duplicate(store_id))
        check('required_tool_failure_get_tid_config', tool_failure(sid,'get_tid_config'))
        check('approval_error_exists', has_approval_error(store_id))
        check('expected_degraded_no_cause', cause is None and state == 'DEGRADED_REVIEW')
    elif fam == 'S6B':
        check('underlying_duplicate_tid_exists', has_active_duplicate(store_id))
        check('optional_tool_failure_get_support_route', tool_failure(sid,'get_support_route'))
        check('required_tid_tool_not_failed', not tool_failure(sid,'get_tid_config'))
        check('approval_error_exists', has_approval_error(store_id))
        check('expected_ready_duplicate_tid', cause == 'duplicate_tid' and state == 'READY_FOR_REVIEW')
    elif fam == 'S7':
        check('current_active_tids_not_duplicate', not has_active_duplicate(store_id))
        check('incident_time_duplicate_exists', incident_duplicate(store_id, cp['incident_at']))
        check('post_incident_reconfiguration_exists', count('installation_events','store_id=? AND event_type=?',(store_id,'configured'))>0)
        check('approval_error_exists', has_approval_error(store_id))
        check('expected_conflict_no_cause', cause is None and state == 'CONFLICT_REVIEW')
    passed = all(c['passed'] for c in checks)
    validation.append({
        'case_id': sid,
        'scenario_family': fam,
        'scenario_type': cp['scenario_type'],
        'store_id': store_id,
        'expected_state': state,
        'expected_primary_cause': cause,
        'passed': passed,
        'checks': checks,
    })

passed_count=sum(1 for v in validation if v['passed'])
summary = {
    'dataset': 'offline_payment_ops_synthetic_50',
    'total_cases': len(validation),
    'passed_cases': passed_count,
    'failed_cases': len(validation)-passed_count,
    'case_distribution': dict(Counter(cp['scenario_family'] for cp in case_plan)),
    'row_counts': {table: len(rows.get(table, [])) for table in columns},
    'validation_scope': 'Row-level evidence validation: checks that each synthetic case contains the operational facts required for its intended state/cause and avoids obvious conflicting evidence.',
}
(REPORTS/'synthetic_50_validation_report.json').write_text(json.dumps({'summary':summary,'cases':validation}, ensure_ascii=False, indent=2), encoding='utf-8')

with (REPORTS/'synthetic_50_validation_matrix.csv').open('w', newline='', encoding='utf-8') as f:
    fieldnames=['case_id','scenario_family','scenario_type','store_id','expected_state','expected_primary_cause','passed','failed_checks']
    w=csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for v in validation:
        failed='; '.join(c['name'] for c in v['checks'] if not c['passed'])
        w.writerow({k:v.get(k,'') for k in fieldnames[:-1]} | {'failed_checks':failed})

# README + md report
(SYN/'README.md').write_text('''# Offline Payment Ops Synthetic 50 Dataset\n\n이 데이터셋은 StoreOps 결제 장애 triage agent의 데이터 기반 판단을 평가하기 위한 50개 합성 운영 사건입니다.\n\n## 구성\n\n- `synthetic_case_plan_50.json`: 사람이 검토하는 case 설계표\n- `raw/*.csv`: 에이전트 tool이 조회할 운영 raw fact 테이블\n- `../../fixtures/offline_payment_ops_synthetic_50.sqlite3`: raw CSV를 적재한 SQLite fixture\n- `../../golden/offline_payment_ops_cases_50.json`: 평가기가 보는 정답 label\n- `../../evaluation/retrieval_cases_50.json`: RAG retrieval 평가 케이스\n- `../../evaluation/planner_cases_50.json`: planner/tool 선택 평가 케이스\n\n## 분포\n\n- S1 duplicate_tid: 10\n- S2 terminal_identifier_mismatch: 7\n- S3 van_merchant_registration_missing: 7\n- S4 pos_front_connection_issue: 7\n- S5 clarification_required: 7\n- S6A required_tool_failure: 4\n- S6B optional_tool_failure: 3\n- S7 temporal_conflict: 5\n\n## 중요한 원칙\n\n운영 raw CSV/SQLite에는 `expected_primary_cause` 같은 정답 원인을 넣지 않습니다. 정답은 `golden` JSON과 검증 리포트에만 있습니다.\n''', encoding='utf-8')

md = []
md.append('# Synthetic 50 Validation Report\n')
md.append(f"- Total cases: {summary['total_cases']}\n")
md.append(f"- Passed: {summary['passed_cases']}\n")
md.append(f"- Failed: {summary['failed_cases']}\n")
md.append('\n## Case distribution\n')
for k,v in summary['case_distribution'].items(): md.append(f'- {k}: {v}\n')
md.append('\n## Row counts\n')
for k,v in summary['row_counts'].items(): md.append(f'- {k}: {v}\n')
md.append('\n## Validation meaning\n')
md.append('이 검증은 LLM/agent 실행 결과 평가가 아니라, 생성된 raw operational facts가 의도한 원인/상태 label과 논리적으로 일치하는지 확인하는 row-level evidence validation입니다.\n')
md.append('\n## Passed cases by family\n')
for fam in ['S1','S2','S3','S4','S5','S6A','S6B','S7']:
    fam_cases=[v for v in validation if v['scenario_family']==fam]
    md.append(f'- {fam}: {sum(1 for v in fam_cases if v["passed"])}/{len(fam_cases)}\n')
(REPORTS/'synthetic_50_validation_report.md').write_text(''.join(md), encoding='utf-8')

# Manifest with hashes
manifest = {
    'dataset_id': 'offline_payment_ops_synthetic_50',
    'created_for': 'StoreOps Incident Agent synthetic evaluation',
    'total_cases': len(case_plan),
    'files': {},
}
for p in sorted(OUT.rglob('*')):
    if p.is_file() and p.name != 'offline_payment_ops_synthetic_50_manifest.json':
        rel=str(p.relative_to(OUT))
        manifest['files'][rel]={'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
(FIX/'offline_payment_ops_synthetic_50_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')

if args.zip_output:
    zip_path = OUT.parent / f'{OUT.name}_synthetic_50.zip'
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.rglob('*')):
            if p.is_file():
                z.write(p, p.relative_to(OUT))
    print('ZIP', zip_path)
print(json.dumps(summary, ensure_ascii=False, indent=2))
print('OUT', OUT)
