#!/usr/bin/env python3
"""Apply/revert patch SPF tren serving.py cua vLLM (chat completion).

SPF = Shortest-Prefill-First: dat priority = len(prompt_token_ids) thay vi
request.priority (mac dinh 0). Prompt ngan -> priority nho -> uu tien truoc khi
chay voi --scheduling-policy=priority.

Dung:
  python3 patch_serving_priority.py apply    # bat SPF (cho v22)
  python3 patch_serving_priority.py revert   # tra ve goc (cho v21 baseline)
  python3 patch_serving_priority.py status   # xem trang thai

Idempotent, tu tim duong dan file qua import, giu ban goc o <file>.orig.bak.
"""
import sys
import vllm.entrypoints.openai.chat_completion.serving as m

PATH = m.__file__
ORIG_LINE = "                    priority=request.priority,\n"
PATCHED_LINE = (
    "                    priority=(len(prompt_token_ids) "
    "if prompt_token_ids else request.priority),\n"
)


def read():
    with open(PATH) as f:
        return f.read()


def write(content):
    with open(PATH, "w") as f:
        f.write(content)


def status(content=None):
    content = content or read()
    if PATCHED_LINE in content:
        return "PATCHED (SPF on)"
    if ORIG_LINE in content:
        return "ORIGINAL (FCFS/baseline)"
    return "UNKNOWN (khong tim thay dong priority chuan -- kiem tra tay!)"


def main():
    action = sys.argv[1] if len(sys.argv) > 1 else "status"
    content = read()
    st = status(content)
    print(f"file: {PATH}")
    print(f"truoc: {st}")

    if action == "status":
        return

    if action == "apply":
        if PATCHED_LINE in content:
            print("da patched, bo qua")
            return
        assert content.count(ORIG_LINE) == 1, \
            f"expected 1 original priority line, got {content.count(ORIG_LINE)}"
        # backup ban goc 1 lan
        bak = PATH + ".orig.bak"
        try:
            open(bak, "x").write(content)
            print(f"backup goc -> {bak}")
        except FileExistsError:
            pass
        write(content.replace(ORIG_LINE, PATCHED_LINE))
    elif action == "revert":
        if ORIG_LINE in content:
            print("da la ban goc, bo qua")
            return
        assert content.count(PATCHED_LINE) == 1, \
            f"expected 1 patched priority line, got {content.count(PATCHED_LINE)}"
        write(content.replace(PATCHED_LINE, ORIG_LINE))
    else:
        print(f"action khong hop le: {action}")
        raise SystemExit(2)

    print(f"sau : {status()}")


if __name__ == "__main__":
    main()
