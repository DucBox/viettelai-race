#!/usr/bin/env python3
"""DIAGNOSTIC (khong phai config nop hop le -- warm bang noi dung trace, giong
thi nghiem docs 4.3): warm dung cai SYSTEM PROMPT CHUNG cua turn 0 (38,956 ky
tu, ~6.4k token, giong het o ca 20 user) MOT LAN de populate prefix cache,
truoc khi ban full trace. Muc dich: DO TRAN loi ich neu shared prefix duoc
cache 1 lan thay vi 20 lan.

Gui 1 request = [system_msg_chung, {user: "."}], max_tokens=1 -> vLLM prefill
+ commit blocks cua hau het system prompt vao prefix cache. Sau do 19+ request
turn-0 se hit phan nay.
"""
import json
import sys
import httpx

TRACE = "/root/trace-round1.jsonl"
URL = "http://localhost:8000/v1/chat/completions"

rows = [json.loads(l) for l in open(TRACE)]
# request 0 = user 0 turn 0; system message la msg[0], giong het o ca 20 user
sys_msg = rows[0]["body"]["messages"][0]
assert sys_msg["role"] == "system", sys_msg["role"]
print(f"system prompt chung: {len(sys_msg['content'])} ky tu")

body = {
    "model": "Qwen3.5-2B",
    "messages": [sys_msg, {"role": "user", "content": "."}],
    "max_tokens": 1,
    "temperature": 0.0,
    "stream": False,
}
with httpx.Client(timeout=120.0) as c:
    r = c.post(URL, json=body)
    r.raise_for_status()
    j = r.json()
    usage = j.get("usage", {})
    print(f"warmup xong: prompt_tokens={usage.get('prompt_tokens')} "
          f"cached={usage.get('prompt_tokens_details', {})}")
print("prefix cache da duoc populate. Gio chay replay_trace_detailed.py.")
