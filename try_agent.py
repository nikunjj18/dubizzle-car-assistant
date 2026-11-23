import pathlib
import tempfile
import time

from assistant.agent import Agent
from assistant.config import INVENTORY_PATH
from assistant.inventory import Inventory
from assistant.memory import MemoryStore

inv = Inventory(INVENTORY_PATH)
mem = MemoryStore(pathlib.Path(tempfile.mkdtemp()) / "try.db")   # throwaway DB
mem.ensure_user("tester", "Tester")
sid = mem.create_session("tester")
agent = Agent(inv, mem)

for msg in [
    "Hi! Show me some Mercedes SUVs",
    "What's the mileage on the first one? Is there a warranty on it?",
    "Can you write me a Python script to sort a list?",
    "Is it cheaper on dubicars?",
]:
    print(f"\nYOU  > {msg}")
    try:
        out = agent.chat("tester", sid, msg)
    except Exception as e:
        print(f"ERROR> {type(e).__name__}: {str(e)[:600]}")
        break
    for t in out["tool_calls"]:
        print(f"       tool: {t['tool']} {t['arguments'][:80]}")
    print(f"DUBI > {out['reply']}")
    time.sleep(3)   # be gentle with the free tier