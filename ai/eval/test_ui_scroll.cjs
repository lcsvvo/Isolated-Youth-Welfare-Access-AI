const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const source = fs.readFileSync(path.join(__dirname, '..', 'ui_scroll.js'), 'utf8');

function fixture() {
  let clock = 0, next = 0;
  const frames = new Map(), events = new Map();
  const main = {scrollTop: 400, calls: 0,
    getBoundingClientRect: () => ({top: 0}),
    scrollTo({top}) { this.scrollTop = top; this.calls++; },
  };
  const context = vm.createContext({window: {}, performance: {now: () => clock},
    requestAnimationFrame(fn) { frames.set(++next, fn); return next; },
    cancelAnimationFrame(id) { frames.delete(id); },
    document: {
      querySelector: selector => selector === 'main' ? main : null,
      getElementById: id => ['chat-reply-4', 'confirm-summary'].includes(id)
        ? {getBoundingClientRect: () => ({top: 900 - main.scrollTop})} : null,
      addEventListener: (event, fn) => events.set(event, fn),
      removeEventListener: event => events.delete(event),
    },
  });
  return {main, events,
    run(state) { vm.runInContext(source.replace('__SCROLL_STATE__', JSON.stringify(state)), context); },
    tick() { clock += 100; const pending = [...frames.values()]; frames.clear(); pending.forEach(fn => fn()); },
    settle() { for (let i = 0; i < 20; i++) this.tick(); },
  };
}

test('a new conversation turn opens at that turn, and unchanged reruns preserve reading position', () => {
  const f = fixture();
  const state = {page: 'chat', token: 'chat:1:2:7', anchor: 'chat-reply-4'};
  f.run(state); f.settle();
  assert.equal(f.main.scrollTop, 876);
  f.main.scrollTop = 300;
  f.run(state); f.settle();
  assert.equal(f.main.scrollTop, 300);
});

test('manual scrolling cancels automatic settling immediately', () => {
  const f = fixture();
  f.run({page: 'chat', token: 'chat:new', anchor: 'chat-reply-4'});
  f.tick();
  f.events.get('wheel')();
  f.main.scrollTop = 600;
  f.settle();
  assert.equal(f.main.scrollTop, 600);
});

test('opening the conditions summary skips older messages and edits preserve scroll', () => {
  const f = fixture();
  const state = {page: 'confirm', token: 'confirm:1', anchor: 'confirm-summary'};
  f.run(state); f.settle();
  assert.equal(f.main.scrollTop, 876);
  f.main.scrollTop = 1250;
  f.run(state); f.settle();
  assert.equal(f.main.scrollTop, 1250);
});

test('results open at the top, while browsing does not reset its list position', () => {
  const f = fixture();
  f.run({page: 'results', token: 'results:1'}); f.settle();
  assert.equal(f.main.scrollTop, 0);
  f.main.scrollTop = 700;
  f.run({page: 'explore', token: 'explore:2'}); f.settle();
  assert.equal(f.main.scrollTop, 700);
});
