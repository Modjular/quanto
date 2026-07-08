import { CONTRAST_DEFAULT } from './config.js';

// A single shared popover reused for whichever image row is active — napari-style
// per-layer contrast limits. The popover edits the image's display-only window
// (black/white points) via backend.setWindow, which only re-runs the composite
// pass: no feature recompute, no retrain, classifier untouched.

let popoverEl = null;   // the floating panel, created lazily
let activeImgId = null; // image the popover currently targets, or null when closed

/**
 * Opens (or moves) the contrast popover next to the given image's row and binds
 * its controls to that image. Calling it again for the same image toggles it shut.
 * @param {Object} state - Shared app state (reads state.images).
 * @param {string} imgId - Id of the image to edit.
 * @param {HTMLElement} anchorEl - Row element/button to anchor the popover beside.
 */
export function openContrastPopover(state, imgId, anchorEl) {
    // Toggle off if the same row's control is clicked again.
    if (activeImgId === imgId && popoverEl && popoverEl.style.display !== 'none') {
        closeContrastPopover();
        return;
    }

    const img = state.images.find(i => i.id === imgId);
    if (!img) return;

    if (!popoverEl) popoverEl = buildPopover();
    activeImgId = imgId;

    const loInput   = popoverEl.querySelector('.contrast-lo');
    const hiInput   = popoverEl.querySelector('.contrast-hi');
    const loVal     = popoverEl.querySelector('.contrast-lo-val');
    const hiVal     = popoverEl.querySelector('.contrast-hi-val');

    const render = () => {
        loVal.textContent = img.windowLo.toFixed(2);
        hiVal.textContent = img.windowHi.toFixed(2);
        loInput.value = img.windowLo;
        hiInput.value = img.windowHi;
    };

    // Keep lo <= hi (napari clamps the handles so they can't cross).
    loInput.oninput = () => {
        img.windowLo = Math.min(parseFloat(loInput.value), img.windowHi);
        img.backend.setWindow(img.windowLo, img.windowHi);
        render();
    };
    hiInput.oninput = () => {
        img.windowHi = Math.max(parseFloat(hiInput.value), img.windowLo);
        img.backend.setWindow(img.windowLo, img.windowHi);
        render();
    };
    popoverEl.querySelector('.contrast-auto').onclick = () => {
        img.windowLo = CONTRAST_DEFAULT.lo;
        img.windowHi = CONTRAST_DEFAULT.hi;
        img.backend.setWindow(img.windowLo, img.windowHi);
        render();
    };

    render();
    positionPopover(popoverEl, anchorEl);
    popoverEl.style.display = 'block';
}

/** Hides the shared popover, if open. */
export function closeContrastPopover() {
    if (popoverEl) popoverEl.style.display = 'none';
    activeImgId = null;
}

// Builds the popover DOM once and wires dismissal (click-outside / Escape).
function buildPopover() {
    const el = document.createElement('div');
    el.className = 'contrast-popover heavy-panel';
    el.innerHTML = `
        <div class="control-group">
            <span class="control-label">Contrast Limits</span>
            <div class="contrast-slider">
                <input type="range" class="contrast-lo" min="0" max="1" step="0.01" value="0">
                <input type="range" class="contrast-hi" min="0" max="1" step="0.01" value="1">
            </div>
            <div class="contrast-readout">
                <span>Black <b class="contrast-lo-val">0.00</b></span>
                <span>White <b class="contrast-hi-val">1.00</b></span>
            </div>
            <button class="contrast-auto">Auto</button>
        </div>`;
    // Clicks inside the popover must not bubble to the document dismiss handler.
    el.addEventListener('mousedown', (e) => e.stopPropagation());
    document.body.appendChild(el);

    document.addEventListener('mousedown', () => closeContrastPopover());
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') closeContrastPopover();
    });

    return el;
}

// Anchors the popover to the right of its row, flipping left/clamping to stay
// on screen (the sidebar is a fixed-width column, so "right" floats over the viewport).
function positionPopover(el, anchorEl) {
    const rect = anchorEl.getBoundingClientRect();
    el.style.visibility = 'hidden';
    el.style.display = 'block';
    const pw = el.offsetWidth;
    const ph = el.offsetHeight;

    let left = rect.right + 8;
    if (left + pw > window.innerWidth - 8) left = rect.left - pw - 8; // flip to left side
    let top = rect.top;
    if (top + ph > window.innerHeight - 8) top = window.innerHeight - ph - 8;

    el.style.left = Math.max(8, left) + 'px';
    el.style.top  = Math.max(8, top) + 'px';
    el.style.visibility = 'visible';
}
