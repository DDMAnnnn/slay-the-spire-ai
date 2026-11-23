"""Write a portable, offline decision-by-decision replay."""
from html import escape


def name(value):
    return escape(value.replace("Strike_R", "Strike").replace("Defend_R", "Defend")
                  .replace("_", " ").title())


def panel(frame, index):
    state = frame["state"]
    player = state["player"]
    enemies = []
    for m in state["monsters"]:
        intent = name(m["intent"])
        if m["intent"].startswith("ATTACK"):
            intent += f" {m['damage']} × {m['hits']}"
        enemies.append(f"<li><b>{name(m['id'])}</b> · {m['current_hp']}/{m['max_hp']} HP"
                       f" · {m['block']} block · {intent}</li>")
    cards = " ".join(f"<span>{name(c['id'])}{'+' if c['upgrades'] else ''}{' · ' + str(c['cost']) + ' energy' if c['cost'] >= 0 else ''}</span>" for c in state["hand"])
    rows = "".join(f"<tr{' class=chosen' if j == frame['chosen'] else ''}><td>"
                   f"{'✓ ' if j == frame['chosen'] else ''}{name(option)}</td><td>{score:.3f}</td></tr>"
                   for j, (option, score) in enumerate(zip(frame["options"], frame["scores"])))
    return (f"<section {'hidden' if index else ''}><small>TURN {state['turn'] + 1}</small>"
            f"<h2>{name(frame['options'][frame['chosen']])}</h2>"
            f"<p>Ironclad · <b>{player['current_hp']}/{player['max_hp']} HP</b> · "
            f"{player['block']} block · {state['energy']} energy</p><ul>{''.join(enemies)}</ul>"
            f"<h3>Hand</h3><div>{cards}</div><h3>{escape(frame['prompt'])}</h3>"
            f"<table><thead><tr><th>Legal action</th><th>Model score</th></tr></thead><tbody>{rows}</tbody></table></section>")


def write_replay(path, result, frames):
    outcome = "Victory" if result["victory"] else "Defeat"
    title = f"Ironclad vs. {name(result['encounter'])}"
    pages = "".join(panel(f, i) for i, f in enumerate(frames))
    pages += (f"<section hidden><small>COMBAT COMPLETE</small><h2>{outcome}</h2>"
              f"<p>{result['start_hp']} starting HP · {result['hp']}/{result['max_hp']} HP at combat end</p>"
              f"<p>{len(frames)} decisions in {frames[-1]['state']['turn'] + 1} turns.</p></section>")
    html = """<!doctype html><html lang=en><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1"><title>Combat replay</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#182522;color:#f7f5e9;font:16px/1.5 "Segoe UI",sans-serif}main{max-width:770px;margin:auto;padding:28px 20px}h1,h2{font-family:Georgia,serif;line-height:1.12}h1{font-size:clamp(26px,5vw,42px);margin:0}h2{font-size:28px;margin:8px 0 20px}small{letter-spacing:.12em;color:#a7dac5}header p{color:#c2d0c9}nav{display:flex;align-items:center;flex-wrap:wrap;gap:8px;margin:22px 0}button{background:#e4cd98;border:0;border-radius:6px;padding:8px 15px;color:#1b2824;font:inherit;cursor:pointer}button:disabled{opacity:.45}output{margin-left:auto}section{border:1px solid #4a6257;border-radius:12px;background:#23332d;padding:22px}h3{font-size:16px;margin:18px 0 8px}ul{padding-left:20px}span{background:#354b42;border-radius:5px;padding:3px 9px;display:inline-block;margin:2px}table{width:100%;border-collapse:collapse;text-align:left}td,th{padding:5px 8px;border-bottom:1px solid #46554d}td:last-child,th:last-child{text-align:right;font-variant-numeric:tabular-nums}.chosen{color:#ffe19b;font-weight:650}button:focus-visible{outline:3px solid white}
</style><main>"""
    html += (f"<header><h1>{title}</h1><p>Ascension 20 · Seed {result['seed']} · Simulator replay</p></header>"
             "<nav aria-label='Playback'><button id=prev>Previous</button><button id=play>Play</button>"
             "<button id=next>Next</button><output id=pos aria-live=polite></output></nav>" + pages)
    html += """</main><script>
const pages=[...document.querySelectorAll('section')], prev=document.querySelector('#prev'),
next=document.querySelector('#next'), play=document.querySelector('#play'), pos=document.querySelector('#pos');
let i=0,timer=null;
function pause(){clearInterval(timer);timer=null;play.textContent='Play'}
function show(n){pages[i].hidden=true;i=Math.min(pages.length-1,Math.max(0,n));pages[i].hidden=false;
prev.disabled=i===0;next.disabled=i===pages.length-1;pos.textContent=next.disabled?'Result':'Decision '+(i+1)+' / '+(pages.length-1);
if(next.disabled)pause()}
prev.onclick=()=>{pause();show(i-1)};next.onclick=()=>{pause();show(i+1)};
play.onclick=()=>{if(timer){pause();return}if(next.disabled)show(0);timer=setInterval(()=>show(i+1),1800);play.textContent='Pause'};
document.addEventListener('keydown',e=>{if(e.key==='ArrowLeft')prev.click();if(e.key==='ArrowRight')next.click()});
show(0)</script></html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf8")
