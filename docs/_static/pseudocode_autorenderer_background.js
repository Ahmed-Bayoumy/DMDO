document.addEventListener("DOMContentLoaded", function() {
  var renderAll = function() {
     
    pseudocode.renderElement(
    document.getElementById("1"), {
        captionCount: 0,
         lineNumber: true 
    });
 
    pseudocode.renderElement(
    document.getElementById("2"), {
        captionCount: 1,
         lineNumber: true 
    });

    if (typeof MathJax !== 'undefined' && MathJax.typesetPromise) {
      MathJax.typesetPromise();
    }
  };
  if (typeof MathJax !== 'undefined' && MathJax.startup) {
    MathJax.startup.promise.then(renderAll);
  } else {
    renderAll();
  }
});