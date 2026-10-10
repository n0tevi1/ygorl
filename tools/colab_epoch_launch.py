"""Build a durable, single-launch remote epoch command and its recovery query."""

import json


def launch_code(command, directory, identity, env):
    # The claim precedes Popen. A crash in between is ambiguous and must stop,
    # rather than risk spawning a second worker on retry.
    spec = json.dumps(dict(command=command, directory=directory, identity=identity, env=env))
    return f"""import fcntl,json,os,pathlib,subprocess,time
s=json.loads({spec!r})
d=pathlib.Path(s['directory']);d.mkdir(parents=True,exist_ok=True)
receipt=d/'launch.json'
with (d/'launch.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 if receipt.exists():
  r=json.loads(receipt.read_text())
  assert r['identity']==s['identity'] and r['command']==s['command'], 'launch identity mismatch'
  assert r['state']=='launched', 'ambiguous launch claim; do not relaunch'
 else:
  r=dict(identity=s['identity'],command=s['command'],state='claimed')
  temp=d/'launch.tmp';temp.write_text(json.dumps(r));temp.replace(receipt)
  wrapper="import pathlib,subprocess; p=subprocess.run("+repr(s['command'])+"); pathlib.Path("+repr(str(d/'exit'))+").write_text(str(p.returncode))"
  with (d/'stdout.log').open('w') as log:
   p=subprocess.Popen([s['command'][0],'-c',wrapper],env=dict(os.environ,**s['env']),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  r.update(state='launched',pid=p.pid)
  temp.write_text(json.dumps(r));temp.replace(receipt)
print('YGORL_JSON:'+json.dumps(r))
"""


def receipt_code(directory):
    return f"""import pathlib,json
p=pathlib.Path({directory!r})/'launch.json'
print('YGORL_JSON:'+json.dumps(json.loads(p.read_text()) if p.exists() else None))
"""


def validate_receipt(receipt, command, identity):
    if not isinstance(receipt, dict) or (
        receipt.get("state") != "launched"
        or receipt.get("command") != command
        or receipt.get("identity") != identity
        or type(receipt.get("pid")) is not int
        or receipt["pid"] <= 0
    ):
        raise RuntimeError("remote launch not proven; do not relaunch")
    return receipt
