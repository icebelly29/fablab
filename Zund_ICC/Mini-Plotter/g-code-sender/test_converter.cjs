const SvgConverter = require('./SvgConverter.cjs');

const converter = new SvgConverter({
  feedRate: 1000,
  safeZ: 5,
  cutZ: 0,
  tolerance: 0.1
});

const svg = `<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
  <!-- Cubic Bezier -->
  <path d="M 10 10 C 20 50, 40 50, 50 10" />
</svg>`;

try {
    const gcode = converter.convert(svg);
    console.log("Generated G-code:");
    console.log(gcode);
} catch (error) {
    console.error("Error:", error);
}
