/**
 * MathJax configuration for NetworkX MCP Server documentation
 * Enables mathematical notation rendering for graph theory formulas
 */

window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
    packages: ['base', 'ams', 'noerrors', 'noundefined'],
    macros: {
      // Graph theory notation macros
      G: "\\mathcal{G}",
      V: "\\mathcal{V}",
      E: "\\mathcal{E}",
      N: "\\mathcal{N}",
      deg: "\\text{deg}",
      indeg: "\\text{in-deg}",
      outdeg: "\\text{out-deg}",
      dist: "\\text{dist}",
      diam: "\\text{diam}",
      ecc: "\\text{ecc}",

      // Centrality measures
      BC: "\\text{BC}",
      CC: "\\text{CC}",
      EC: "\\text{EC}",
      PR: "\\text{PR}",

      // Common mathematical notations
      argmax: "\\text{argmax}",
      argmin: "\\text{argmin}",
      bigO: "\\mathcal{O}",

      // Set operations
      Union: "\\bigcup",
      Intersection: "\\bigcap",

      // Probability
      Prob: "\\mathbb{P}",
      Expect: "\\mathbb{E}",
      Var: "\\text{Var}",

      // Graph specific functions
      clustering: "\\text{clustering}",
      modularity: "\\text{modularity}",
      pagerank: "\\text{PageRank}",
      betweenness: "\\text{betweenness}",
      closeness: "\\text{closeness}",
      eigenvector: "\\text{eigenvector}",

      // Matrix notation
      A: "\\mathbf{A}",
      D: "\\mathbf{D}",
      L: "\\mathbf{L}",
      I: "\\mathbf{I}",

      // Common symbols
      infty: "\\infty",
      approx: "\\approx",
      propto: "\\propto",
      sim: "\\sim",

      // Complexity notation
      On: ["\\mathcal{O}(#1)", 1],
      Omegan: ["\\Omega(#1)", 1],
      Thetan: ["\\Theta(#1)", 1]
    }
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  },
  loader: {
    load: ['[tex]/noerrors']
  }
};

document$.subscribe(() => {
  MathJax.typesetPromise()
});

// Add custom styling for math blocks
document.addEventListener('DOMContentLoaded', function() {
  // Add custom classes to math elements for styling
  const mathElements = document.querySelectorAll('.MathJax');
  mathElements.forEach(element => {
    element.classList.add('math-formula');
  });

  // Add copy functionality to math formulas
  const displayMathElements = document.querySelectorAll('.MathJax_Display');
  displayMathElements.forEach(element => {
    element.addEventListener('click', function() {
      const mathML = this.querySelector('math');
      if (mathML) {
        const latex = mathML.getAttribute('alttext') || 'Formula';
        navigator.clipboard.writeText(latex).then(() => {
          // Show tooltip or notification
          showCopyNotification(this);
        });
      }
    });
  });
});

function showCopyNotification(element) {
  const notification = document.createElement('div');
  notification.textContent = 'Formula copied!';
  notification.style.cssText = `
    position: absolute;
    background: #333;
    color: white;
    padding: 4px 8px;
    border-radius: 4px;
    font-size: 12px;
    z-index: 1000;
    pointer-events: none;
    opacity: 0;
    transition: opacity 0.3s;
  `;

  element.appendChild(notification);

  // Position the notification
  const rect = element.getBoundingClientRect();
  notification.style.left = '50%';
  notification.style.top = '-30px';
  notification.style.transform = 'translateX(-50%)';

  // Show and hide animation
  setTimeout(() => notification.style.opacity = '1', 10);
  setTimeout(() => {
    notification.style.opacity = '0';
    setTimeout(() => element.removeChild(notification), 300);
  }, 2000);
}
