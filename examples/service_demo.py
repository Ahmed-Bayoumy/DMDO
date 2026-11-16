"""The DMDO HTTP service end to end, on one machine: a service, two worker processes and a client.

On real hardware the three parts run in different places:

    dmdo serve  --host 0.0.0.0 --port 8400 --token $DMDO_TOKEN            # service host
    dmdo worker --connect http://dmdo-host:8400 --tags aero --token ...   # machine with the aero code
    dmdo worker --connect http://dmdo-host:8400 --tags structures ...     # machine with the structures code

and the client below runs anywhere (any language can use the HTTP API instead; see DMDO.service.server).

Run from this directory:  python service_demo.py
"""
import os
import secrets
import subprocess
import sys

from DMDO import load_setup
from DMDO.service import Client, start_service

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
  token = secrets.token_urlsafe(16)
  service = start_service(root_dir=os.path.join(HERE, "_runs", "service"), token=token)
  print(f"service at {service.url}")

  # Two workers in their own processes; --path makes basic_disciplines importable for them.
  workers = [subprocess.Popen([sys.executable, "-m", "DMDO.service", "worker", "--connect", service.url,
                               "--tags", tag, "--token", token, "--path", HERE, "--name", tag])
             for tag in ("aero", "structures")]
  try:
    client = Client(service.url, token=token)
    setup = load_setup(os.path.join(HERE, "basic_problem.yaml"))
    setup["coord"]["c1"]["budget"] = 8
    setup["MDO"]["display"] = False
    session = client.create_session(
      setup=setup, mdo_name="demo",
      blackboxes={"A1": "basic_disciplines:A1", "A2": "basic_disciplines:A2",
                  "opt1": "basic_disciplines:opt1", "opt2": "basic_disciplines:opt2"},
      sweep="jacobi", placement={1: "aero", 2: "structures"}, timeout=300)

    session.run()                                   # runs in the service; solves go to the workers
    for event in session.events():                  # live progress (server-sent events)
      if event["event"] == "iteration_end":
        d = event["data"]
        print(f"iteration {d['iteration']}: objective {d['objective']:.6f}  max|q| {d['max_inconsistency']:.3e}")
      else:
        print(event["event"], event["data"])

    result = session.result()
    print("status:", result["status"], "| variables:", result["variables"])
  finally:
    for w in workers:
      w.terminate()
    service.shutdown()


if __name__ == "__main__":
  main()
