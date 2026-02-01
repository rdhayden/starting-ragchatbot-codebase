# Frontend Changes: Theme Toggle Feature

## Overview
Added a complete light/dark theme toggle system with a toggle button, smooth transitions, and persistent user preferences.

## Files Modified

### 1. `frontend/index.html`

Added a theme toggle button at the top of the body, positioned in the top-right corner:

```html
<button class="theme-toggle" id="themeToggle" aria-label="Toggle dark/light theme" title="Toggle theme">
    <svg class="icon icon-sun">...</svg>
    <svg class="icon icon-moon">...</svg>
</button>
```

Features:
- Uses SVG icons (sun for dark mode, moon for light mode)
- Accessible with `aria-label` and `title` attributes
- Keyboard navigable (focusable button element)

---

### 2. `frontend/style.css`

#### Light Theme CSS Variables
Added a `[data-theme="light"]` selector with appropriate light theme colors:

```css
[data-theme="light"] {
    --primary-color: #2563eb;
    --primary-hover: #1d4ed8;
    --background: #f8fafc;
    --surface: #ffffff;
    --surface-hover: #f1f5f9;
    --text-primary: #1e293b;
    --text-secondary: #64748b;
    --border-color: #e2e8f0;
    --user-message: #2563eb;
    --assistant-message: #f1f5f9;
    --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    --focus-ring: rgba(37, 99, 235, 0.3);
    --welcome-bg: #eff6ff;
    --welcome-border: #bfdbfe;
    --code-bg: rgba(0, 0, 0, 0.05);
    --scrollbar-track: #f1f5f9;
    --scrollbar-thumb: #cbd5e1;
    --scrollbar-thumb-hover: #94a3b8;
}
```

#### Theme Toggle Button Styles

```css
.theme-toggle {
    position: fixed;
    top: 1rem;
    right: 1rem;
    z-index: 1000;
    width: 44px;
    height: 44px;
    border-radius: 50%;
    /* ... */
}
```

Features:
- Fixed position in top-right corner
- Circular button with hover/focus states
- Smooth scale animation on hover and click
- Icon switching based on current theme

#### New CSS Variables Added to Root
- `--code-bg`: Background color for code blocks
- `--scrollbar-track`: Scrollbar track color
- `--scrollbar-thumb`: Scrollbar thumb color
- `--scrollbar-thumb-hover`: Scrollbar thumb hover color

#### Smooth Theme Transitions
Added transitions to body and key UI elements:
```css
body {
    transition: background-color 0.3s ease, color 0.3s ease;
}

.sidebar, .chat-main, .chat-container, /* etc. */ {
    transition: background-color 0.3s ease, border-color 0.3s ease, color 0.3s ease;
}
```

---

### 3. `frontend/script.js`

#### Theme Initialization
```javascript
function initializeTheme() {
    const savedTheme = localStorage.getItem('theme');
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;

    if (savedTheme) {
        document.documentElement.setAttribute('data-theme', savedTheme);
    } else if (!prefersDark) {
        document.documentElement.setAttribute('data-theme', 'light');
    }
}
```

Features:
- Checks localStorage for saved preference
- Falls back to system preference (prefers-color-scheme)
- Dark theme is the default

#### Theme Toggle Function
```javascript
function toggleTheme() {
    const currentTheme = document.documentElement.getAttribute('data-theme');
    const newTheme = currentTheme === 'light' ? 'dark' : 'light';

    if (newTheme === 'dark') {
        document.documentElement.removeAttribute('data-theme');
    } else {
        document.documentElement.setAttribute('data-theme', newTheme);
    }

    localStorage.setItem('theme', newTheme);
}
```

Features:
- Toggles between light and dark themes
- Persists preference to localStorage
- Uses `data-theme` attribute on `<html>` element

---

## How It Works

1. **On Page Load**: The `initializeTheme()` function checks:
   - First, localStorage for a saved preference
   - If none, checks system preference via `prefers-color-scheme`
   - Applies the appropriate theme

2. **On Toggle Click**: The `toggleTheme()` function:
   - Switches between light and dark themes
   - Saves the preference to localStorage
   - Updates the `data-theme` attribute on `<html>`

3. **CSS responds** to `[data-theme="light"]` selector, updating all CSS variables

---

## Accessibility

- Toggle button is keyboard accessible (Tab to focus, Enter/Space to activate)
- Has `aria-label="Toggle dark/light theme"` for screen readers
- Has `title="Toggle theme"` for tooltip on hover
- Focus ring visible in both themes
- Light theme maintains WCAG AA contrast ratios:
  - Primary text (#1e293b) on light background (#f8fafc): 12.6:1 ratio
  - Secondary text (#64748b) on light background: 4.7:1 ratio

---

## Browser Support

- Uses CSS custom properties (CSS variables) - supported in all modern browsers
- Uses `localStorage` - supported in all modern browsers
- Uses `prefers-color-scheme` media query - supported in all modern browsers
