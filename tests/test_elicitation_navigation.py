"""Execute the real sidebar handlers with a minimal DOM (no server mutations)."""

import shutil
import subprocess
from pathlib import Path

import pytest


def test_sidebar_navigation_is_read_only_and_returnable() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for UI handler tests")
    ui = Path(__file__).parents[1] / "src/genesis/static/ui.html"
    result = subprocess.run(
        [node, "-", str(ui)],
        input=r"""
const fs = require('fs'), vm = require('vm'), assert = require('assert/strict');
const elements = new Map();
function element() {
  return {style:{}, children:[], textContent:'', value:'', disabled:false,
    classList:{toggle(){},remove(){},add(){},contains(){return false;}},
    appendChild(child){this.children.push(child);},
    querySelector(selector){
      for (const child of this.children) {
        if ((child.className || '').split(' ').includes(selector.slice(1))) return child;
        const found = child.querySelector(selector); if (found) return found;
      }
      return null;
    },
    replaceChildren(){this.children=[];}, setAttribute(){},
    set innerHTML(value){this.children=[];}, get innerHTML(){return '';}};
}
global.document = {
  getElementById(id){if(!elements.has(id)) elements.set(id, element()); return elements.get(id);},
  createElement(){return element();}, querySelectorAll(){return [];},
  querySelector(selector){return this.getElementById(selector);}
};
const html = fs.readFileSync(process.argv[2], 'utf8');
global.location = {hash:''};
global.history = {replaceState(){}};
vm.runInThisContext(html.match(/<script>([\s\S]*?)<\/script>/)[1]);
vm.runInThisContext(`
elicitationSessionId = 'saved';
elicitationWorkflow = {stages:[
  {id:'openness',title:'Layer 1'}, {id:'theory',title:'Layer 2'},
  {id:'domain',title:'Domain'}]};
elicitationSession = {current_stage:'theory',status:'awaiting_answer',
  allowed_actions:['submit_message','reopen_stage'],
  stages:{openness:{status:'approved',turn_count:6,accepted_revision:3},
    theory:{status:'clarifying',turn_count:2},domain:{status:'not_started'}},
  turns:[{stage_id:'openness',question:'What remains open?',
    answer:'Approved answer',id:1,response_mode:'free_form'},
    {stage_id:'theory',question:'Explain the theory.',answer:'Theory answer',
    id:2,response_mode:'free_form'}]};
`);
let requests = 0;
request = async () => {requests++; throw new Error('Navigation must not send requests');};
const before = vm.runInThisContext('JSON.stringify(elicitationSession)');
document.getElementById('elicitation-composer').value = 'Unsent Layer 2 answer';
renderStageRail(vm.runInThisContext('elicitationSession'));
const buttons = document.getElementById('elicitation-stage-list').children;
assert.equal(buttons[1].disabled, false, 'Current stage must remain reachable');
assert.equal(buttons[2].disabled, true, 'Unstarted stage is not navigable');
buttons[0].onclick();
assert.equal(requests, 0);
assert.equal(document.getElementById('active-stage-content').hidden, true);
assert.match(document.getElementById('stage-history-status').textContent, /6/);
const saved = document.getElementById('stage-history-transcript');
assert.equal(saved.children.length, 2, 'History must show question then answer');
function message(node) {
  return [node.querySelector('.transcript-speaker').textContent,
    node.querySelector('.transcript-bubble').textContent];
}
assert.deepEqual(saved.children.map(message),
  [['Assistant', 'What remains open?'], ['You', 'Approved answer']]);
document.getElementById('elicitation-stage-list').children[1].onclick();
assert.equal(document.getElementById('active-stage-content').hidden, false);
assert.equal(document.getElementById('stage-history-view').hidden, true);
assert.equal(document.getElementById('elicitation-composer').value, 'Unsent Layer 2 answer');
assert.equal(vm.runInThisContext('JSON.stringify(elicitationSession)'), before);
renderElicitation(vm.runInThisContext('elicitationSession'));
assert.deepEqual(document.getElementById('elicitation-transcript').children.map(message),
  [['Assistant', 'Explain the theory.'], ['You', 'Theory answer']]);
// Missing legacy questions must not be invented; text must not become HTML.
const legacy = element();
renderConversationTurns(legacy, [
  {id:3, answer:'Legacy answer'},
  {id:4, question:'Line one\n<script>not markup</script>', answer:'Line one\nLine two'}]);
assert.deepEqual(legacy.children.map(message), [['You','Legacy answer'],
  ['Assistant','Line one\n<script>not markup</script>'], ['You','Line one\nLine two']]);
assert.equal(legacy.children[1].querySelector('.transcript-bubble').children.length, 0);
assert.match(html, /\.transcript-bubble\s*\{[^}]*white-space:\s*pre-wrap/);
global.confirm = () => false;
reopenElicitation('openness').then(() => {
  assert.equal(requests, 0, 'Declining revision confirmation must not mutate');
});
""",
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
