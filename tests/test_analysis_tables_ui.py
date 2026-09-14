"""The Outputs tab shows the analysis tables and exports an experiment.

Datasets and the experiment export existed only behind the API and CLI, so a
researcher using the page could not see the tables their study produced.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from genesis.app import create_app
from tests.test_dataset_export import STUDY

UI = Path(__file__).parents[1] / "src/genesis/static/ui.html"

HARNESS = r"""
const fs = require('fs'), vm = require('vm'), assert = require('assert/strict');
const elements = new Map();
function element() {
  return {style:{}, children:[], textContent:'', value:'', disabled:false, dataset:{}, title:'',
    classList:{toggle(){},remove(){},add(){},contains(){return false;}},
    appendChild(child){this.children.push(child);},
    get options(){return this.children;},
    set innerHTML(value){this.children=[];}, get innerHTML(){return '';}};
}
global.document = {
  getElementById(id){if(!elements.has(id)) elements.set(id, element()); return elements.get(id);},
  createElement(){return element();}, querySelectorAll(){return [];},
};
global.location = {hash:''};
const html = fs.readFileSync(process.argv[2], 'utf8');
vm.runInThisContext(html.match(/<script>([\s\S]*?)<\/script>/)[1]);
const calls = [];
const title = '<img src=x onerror=alert(1)>';
request = async (url, options = {}) => {
  calls.push({url, options});
  if (url.includes('/datasets/')) {
    const offset = Number(new URL('http://x' + url).searchParams.get('offset'));
    return {dataset:'articles', total:3, offset, limit:2, columns:['phase','title'],
            rows: offset === 0
              ? [{phase:1, title}, {phase:2, title:null}]
              : [{phase:3, title:'c'}]};
  }
  if (url.includes('/datasets')) {
    return {datasets:{articles:{rows:3, columns:{phase:{}, title:{}}}}};
  }
  if (url === '/experiments') return [{id:'exp-1', study_id:'s'}];
  if (url.includes('/export')) return {experiment_id:'exp-1', status:'exported', paths:['a']};
  throw new Error('unexpected ' + url);
};
document.getElementById('run-select').value = 'run-7';
(async () => {
  document.getElementById('dataset-limit').value = '2';
  document.getElementById('analysis-build').value = 'builds/p9';
  await loadDatasets();
  assert.match(calls[0].url, /^\/runs\/run-7\/datasets\?analysis_build=builds%2Fp9$/);
  const select = document.getElementById('dataset-select');
  assert.equal(select.children.length, 1);
  select.value = 'articles';
  await loadDatasetPage(0);
  const table = document.getElementById('dataset-table');
  assert.equal(table.children.length, 3, 'a header row and two data rows');
  assert.equal(table.children[1].children[1].textContent, title, 'model text is shown as text');
  assert.equal(table.children[2].children[1].textContent, '', 'a null cell is empty');
  assert.match(document.getElementById('dataset-status').textContent, /rows 1–2 of 3/);
  assert.equal(document.getElementById('dataset-prev').disabled, true);
  assert.equal(document.getElementById('dataset-next').disabled, false);
  datasetPage(1);
  await new Promise(resolve => setTimeout(resolve, 0));
  assert.match(calls.at(-1).url, /offset=2&limit=2&analysis_build=builds%2Fp9$/);
  assert.equal(document.getElementById('dataset-next').disabled, true);
  noteRunSelection();
  assert.equal(document.getElementById('dataset-table').children.length, 0,
               'a table from one run is cleared when the run changes');
  await refreshExperiments();
  document.getElementById('experiment-select').value = 'exp-1';
  await exportExperiment();
  const exported = calls.at(-1);
  assert.equal(exported.url, '/experiments/exp-1/export');
  assert.deepEqual(JSON.parse(exported.options.body),
                   {output:'exports/exp-1-tables', analysis_build:'builds/p9'});
  console.log('ok');
})().catch(error => { console.error(error); process.exit(1); });
"""


def test_the_tables_and_the_experiment_export_are_driven_from_the_page() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for UI handler tests")
    result = subprocess.run(
        [node, "-", str(UI)], input=HARNESS, capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr or result.stdout
    assert "ok" in result.stdout


def test_the_dataset_routes_serve_what_the_export_writes(tmp_path: Path) -> None:
    client = TestClient(create_app(tmp_path))
    service = client.app.state.service
    draft = service.create_specification(STUDY)
    service.approve_specification("tables", draft["version"], "researcher")
    build = service.compile_study(None, "builds/tables", specification_id="tables")["path"]
    service.create_run({"id": "run", "study_id": "tables", "build": build})
    service.execute_run("run", executor_overrides={"tick": lambda _inv: {}})

    listed = client.get("/runs/run/datasets").json()
    assert listed["datasets"]["rounds"]["rows"] == 3
    page = client.get("/runs/run/datasets/rounds?offset=1&limit=1").json()
    assert page["total"] == 3 and len(page["rows"]) == 1 and page["rows"][0]["phase"] == 2
    service.export_run("run", "exports/run")
    exported = (service.workspace / "exports/run/datasets/rounds.csv").read_text().splitlines()
    assert exported[0].split(",") == page["columns"]
    assert client.get("/runs/run/datasets/nope").status_code == 404
    assert client.get("/runs/run/datasets/rounds?limit=0").status_code == 422
    assert json.loads(json.dumps(page))  # the page is plain JSON
