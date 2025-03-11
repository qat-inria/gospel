export default function createBrickworkStateGraph(data) {
    const svgNS = "http://www.w3.org/2000/svg";
    // Configuration for node appearance and spacing.
    const nodeRadius = 22;
    const spacingX = 55;
    const spacingY = 55;
    const offsetX = 24;
    const offsetY = 24;
  
    const numRows = data[0].length;
    const numCols = data.length;
  
    // Calculate the size of the SVG canvas.
    const svgWidth = offsetX * 2 + (numCols - 1) * spacingX;
    const svgHeight = offsetY * 2 + (numRows - 1) * spacingY;
  
    // Create the main SVG element.
    const svg = document.createElementNS(svgNS, "svg");
    svg.setAttribute("width", svgWidth);
    svg.setAttribute("height", svgHeight);
  
    // Draw connecting lines (edges) between adjacent nodes.
    // Horizontal edges: connect each node with its right neighbor.
    for (let i = 0; i < numRows; i++) {
        for (let j = 0; j < numCols - 1; j++) {
            const x1 = offsetX + j * spacingX;
            const y1 = offsetY + i * spacingY;
            const x2 = offsetX + (j + 1) * spacingX;
            const y2 = y1;
            const line = document.createElementNS(svgNS, "line");
            line.setAttribute("x1", x1);
            line.setAttribute("y1", y1);
            line.setAttribute("x2", x2);
            line.setAttribute("y2", y2);
            line.setAttribute("stroke", "#000");
            svg.appendChild(line);
        }
    }
  
    // Vertical edges: connect each node with its bottom neighbor.
    for (let i = 0; i < numRows - 1; i++) {
      for (let j = 0; j < numCols; j++) {
          if (j > 0 && (j % 4 == 2 || j % 4 == 0) && (i % 2 == Math.floor((j - 1) / 4) % 2)) {
              const x1 = offsetX + j * spacingX;
              const y1 = offsetY + i * spacingY;
              const x2 = x1;
              const y2 = offsetY + (i + 1) * spacingY;
              const line = document.createElementNS(svgNS, "line");
              line.setAttribute("x1", x1);
              line.setAttribute("y1", y1);
              line.setAttribute("x2", x2);
              line.setAttribute("y2", y2);
              line.setAttribute("stroke", "#000");
              svg.appendChild(line);
          }
      }
    }
  
    // Create nodes as circles with text labels.
    for (let i = 0; i < numRows; i++) {
        for (let j = 0; j < numCols; j++) {
            const cx = offsetX + j * spacingX;
            const cy = offsetY + i * spacingY;
           
            // Create the circle.
            const circle = document.createElementNS(svgNS, "circle");
            circle.setAttribute("cx", cx);
            circle.setAttribute("cy", cy);
            circle.setAttribute("r", nodeRadius);
            circle.setAttribute("fill", "#add8e6");
            circle.setAttribute("stroke", "#000");
            svg.appendChild(circle);
           
            // Create the text label.
            const text = document.createElementNS(svgNS, "text");
            text.setAttribute("x", cx);
            // Adjust y position slightly for vertical centering.
            text.setAttribute("y", cy + 2);
            text.setAttribute("text-anchor", "middle");
            text.setAttribute("font-size", "14");
            text.textContent = data[j][i];
            svg.appendChild(text);
        }
    }
  
    return svg;
}
