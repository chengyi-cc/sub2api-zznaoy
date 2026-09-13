const center = { x: 300, y: 292 }
const tilt = Math.PI * 0.14
const rotation = -Math.PI * 0.19

function ribbonPoint(angle, offset) {
  const twist = angle + Math.PI * 0.22
  const radius = 190 + offset * 43 * Math.cos(twist)
  const horizontal = radius * Math.cos(angle)
  const vertical = radius * Math.sin(angle) * Math.cos(tilt) - offset * 43 * Math.sin(twist) * Math.sin(tilt)
  return {
    x: center.x + horizontal * Math.cos(rotation) - vertical * Math.sin(rotation),
    y: center.y + horizontal * Math.sin(rotation) + vertical * Math.cos(rotation),
  }
}

function closedSpline(points) {
  const number = value => value.toFixed(2)
  const commands = [`M${number(points[0].x)} ${number(points[0].y)}`]
  for (let index = 0; index < points.length; index += 1) {
    const previous = points[(index + points.length - 1) % points.length]
    const current = points[index]
    const next = points[(index + 1) % points.length]
    const after = points[(index + 2) % points.length]
    const firstControl = { x: current.x + (next.x - previous.x) / 6, y: current.y + (next.y - previous.y) / 6 }
    const secondControl = { x: next.x - (after.x - current.x) / 6, y: next.y - (after.y - current.y) / 6 }
    commands.push(`C${number(firstControl.x)} ${number(firstControl.y)} ${number(secondControl.x)} ${number(secondControl.y)} ${number(next.x)} ${number(next.y)}`)
  }
  return `${commands.join(' ')}Z`
}

export const goldThreads = Array.from({ length: 18 }, (_, index) => {
  const offset = (index / 17) * 2 - 1
  const points = Array.from({ length: 48 }, (_, sample) => ribbonPoint((sample / 48) * Math.PI * 2, offset))
  return {
    points,
    path: closedSpline(points),
    width: index === 0 || index === 17 ? 0.85 : 0.65,
    opacity: Number((0.28 + 0.43 * Math.sin((index / 17) * Math.PI)).toFixed(3)),
  }
})

export const goldArt = `<div class="hc-gold-art" role="img" aria-label="香槟金细丝轻盈交织成环，中央悬置精细字母标记"><svg viewBox="0 0 600 590" fill="none" aria-hidden="true"><defs><linearGradient id="hc-gold-silk" x1="122" y1="92" x2="455" y2="509" gradientUnits="userSpaceOnUse"><stop stop-color="#776344"/><stop offset=".24" stop-color="#dbc89e"/><stop offset=".43" stop-color="#a68a59"/><stop offset=".62" stop-color="#64543b"/><stop offset=".83" stop-color="#e6d3aa"/><stop offset="1" stop-color="#8b7550"/></linearGradient><linearGradient id="hc-gold-seal" x1="265" y1="251" x2="328" y2="337" gradientUnits="userSpaceOnUse"><stop stop-color="#ead9b3"/><stop offset=".48" stop-color="#9d875e"/><stop offset="1" stop-color="#d8c396"/></linearGradient><radialGradient id="hc-gold-air"><stop stop-color="#c5a465" stop-opacity=".055"/><stop offset=".72" stop-color="#c5a465" stop-opacity=".018"/><stop offset="1" stop-color="#c5a465" stop-opacity="0"/></radialGradient></defs><circle class="hc-gold-air" cx="300" cy="292" r="239" fill="url(#hc-gold-air)"/><g class="hc-gold-ribbon" stroke="url(#hc-gold-silk)" stroke-linecap="round" stroke-linejoin="round">${goldThreads.map(thread => `<path class="hc-gold-thread" d="${thread.path}" stroke-width="${thread.width}" stroke-opacity="${thread.opacity}"/>`).join('')}</g><g class="hc-gold-signature"><circle cx="300" cy="292" r="48" stroke="url(#hc-gold-seal)" stroke-width=".65" stroke-opacity=".43"/><path d="M255 273A49 49 0 0 1 313 245M345 311A49 49 0 0 1 287 339" stroke="url(#hc-gold-seal)" stroke-width=".6" stroke-opacity=".15"/><path d="M285 308V276H293L309 308H315V276" stroke="url(#hc-gold-seal)" stroke-width="1.5" stroke-linecap="square" stroke-linejoin="miter"/><path d="M297 354H303" stroke="#bda678" stroke-width=".65" stroke-opacity=".5"/></g></svg><span class="hc-gold-art-label">THE ART OF CONNECTION</span><span class="hc-gold-coordinate">01 — ∞ / AN OPEN WORLD</span></div>`
