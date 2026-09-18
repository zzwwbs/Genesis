"""The Run tab can pause, resume and cancel a run, and says why a run paused."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

UI = Path(__file__).parents[1] / "src/genesis/static/ui.html"

HARNESS = r"""
const fs = require('fs'), vm = require('vm'), assert = require('assert/strict');
const elements = new Map();
function element() {
  return {style:{}, children:[], textContent:'', value:'', disabled:false, dataset:{},
    classList:{toggle(){},remove(){},add(){},contains(){return false;}},
    appendChild(child){this.children.push(child);}, get options(){return this.children;},
    set innerHTML(value){this.children=[];}, get innerHTML(){return '';}};
}
global.document = {
  getElementById(id){if(!elements.has(id)) elements.set(id, element()); return elements.get(id);},
  createElement(){return element();}, querySelectorAll(){return [];},
};
global.location = {hash:''};
global.confirm = () => true;
const html = fs.readFileSync(process.argv[2], 'utf8');
vm.runInThisContext(html.match(/<script>([\s\S]*?)<\/script>/)[1]);
const calls = [];
let status = 'paused';
request = async (url, options = {}) => {
  calls.push({url, options});
  if (url === '/runs') return [{id:'done', status:'completed'}, {id:'r1', status}];
  const error = 'PROVIDER_HTTP: provider returned HTTP 402: Insufficient Balance';
  if (url === '/runs/r1') {
    return {id:'r1', status, version:7,
            executions:[{paused_by:{kind:'provider_credit', status:402, error}}]};
  }
  if (url === '/runs/r1/resume') { status = 'completed'; return {id:'r1', status}; }
  if (url === '/runs/r1/pause') { status = 'paused'; return {}; }
  if (url === '/runs/r1/cancel') { status = 'cancelled'; return {}; }
  throw new Error('unexpected ' + url);
};
(async () => {
  await refreshControlRuns();
  const select = document.getElementById('control-run');
  assert.deepEqual(select.children.map(o => o.value), ['r1'],
                   'only runs that can still act are offered');
  select.value = 'r1';
  await showRunControl();
  const shown = document.getElementById('control-output').textContent;
  assert.match(shown, /provider_credit \(HTTP 402\)/);
  assert.equal(document.getElementById('control-resume').disabled, false);
  assert.equal(document.getElementById('control-pause').disabled, true);
  await resumeControlledRun();
  const resume = calls.find(c => c.url === '/runs/r1/resume');
  assert.equal(resume.options.method, 'POST');
  assert.ok(resume.options.timeoutMs >= 60 * 60 * 1000, 'a resume runs as long as the run does');
  status = 'running';
  await showRunControl();
  assert.equal(document.getElementById('control-pause').disabled, false);
  await pauseControlledRun();
  const pause = calls.find(c => c.url === '/runs/r1/pause');
  assert.equal(pause.options.headers['If-Match'], '"7"');
  await cancelControlledRun();
  assert.ok(calls.some(c => c.url === '/runs/r1/cancel'));
  console.log('ok');
})().catch(error => { console.error(error); process.exit(1); });
"""


def test_the_run_tab_pauses_resumes_and_cancels() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for UI handler tests")
    result = subprocess.run(
        [node, "-", str(UI)], input=HARNESS, capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr or result.stdout
    assert "ok" in result.stdout
