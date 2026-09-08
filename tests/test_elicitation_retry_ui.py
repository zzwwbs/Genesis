"""Exercise real mutation handlers after a response is lost in transit."""

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "handler",
    [
        "sendElicitation()",
        "draftElicitation()",
        "approveElicitation()",
        "saveDraftEdit()",
        "reviseElicitation()",
        "reopenElicitation('openness')",
        "cancelElicitation()",
    ],
)
def test_retry_preserves_original_mutation(handler: str) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for UI handler tests")
    ui = Path(__file__).parents[1] / "src/genesis/static/ui.html"
    result = subprocess.run(
        [node, "-", str(ui), handler],
        input=r"""
const fs = require('fs'), vm = require('vm'), assert = require('assert/strict');
const elements = new Map();
global.document = {
  getElementById(id) {
    if (!elements.has(id)) elements.set(id, {value:'original', style:{}, focus(){},
      classList:{add(){},remove(){},toggle(){},contains(){return false;}}});
    return elements.get(id);
  }, querySelectorAll(){return [];}, querySelector(){return this.getElementById('query');}
};
global.location = {hash:''}; global.history = {replaceState(){}};
global.confirm = () => true;
const html = fs.readFileSync(process.argv[2], 'utf8');
vm.runInThisContext(html.match(/<script>([\s\S]*?)<\/script>/)[1]);
renderElicitation = () => {};
elicitationRequestTimeout = async () => 180000;
vm.runInThisContext(`elicitationSessionId='original-session'; elicitationVersion=7;
  elicitationSession={model_profile_id:'profile'}; selectedSuggestionIndex=1;`);
const attempts = [];
request = async (url, options) => {
  attempts.push({url, body:options.body, key:options.headers['Idempotency-Key']});
  if (attempts.length === 1) throw new Error('Response lost after commit');
  return {};
};
(async () => {
  await vm.runInThisContext(process.argv[3]);
  assert.equal(attempts.length, 1);
  vm.runInThisContext(`elicitationSessionId='different-session';
    elicitationVersion=99; selectedSuggestionIndex=2;`);
  for (const el of elements.values()) el.value = 'edited since failure';
  await retryElicitationAction();
  assert.equal(attempts.length, 2);
  assert.deepEqual(attempts[1], attempts[0], 'Retry must replay the exact original request');
  assert.ok(attempts[0].key);
})().catch(error => {console.error(error); process.exitCode=1;});
""",
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
