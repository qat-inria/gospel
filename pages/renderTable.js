import drawQASMCircuit from "qasmViewer";
import createBrickworkStateGraph from "brickworkStateGraph";

// Global variables to hold table data and sort state.
let tableData = [];
let currentSortColumn = -1; // -1 means no column sorted yet
let currentSortDirection = 1; // 1 for ascending, -1 for descending

// Function to render the table using tableData array.
function createFigure(contents, caption) {
    const figure = document.createElement("figure");
    figure.appendChild(contents);
    const figcaption = document.createElement("figcaption");
    figure.appendChild(figcaption);
    figcaption.appendChild(document.createTextNode(caption))
    return figure;
}
function renderTable(circuits_dirname, brickwork_state_table_dirname) {
    const tbody = document.querySelector('#data-table tbody');
    tbody.innerHTML = ""; // Clear existing rows
  
    tableData.forEach(([circuit, prob]) => {
        const row = document.createElement('tr');
        const keyCell = document.createElement('td');
        const name_div = document.createElement("div");
        keyCell.appendChild(name_div);
        name_div.classList.add("name");
        name_div.textContent = circuit;
        let contents = null;
        name_div.addEventListener("click", function () {
            if (contents == null) {
                contents = document.createElement("div");
                keyCell.appendChild(contents);
                const download_div = document.createElement("div");
                contents.appendChild(download_div);
                download_div.appendChild(document.createTextNode("Download QASM file: "));
                const download_a = document.createElement("a");
                download_div.appendChild(download_a);
                const circuit_url = `${circuits_dirname}/${circuit}`;
                download_a.setAttribute("href", circuit_url)
                download_a.appendChild(document.createTextNode(circuit));
                const json_filename = circuit.replace(/\.\w+$/, ".json");
                const circuit_div = document.createElement("div");
                download(circuit_url, function (circuit_file) {
                    drawQASMCircuit(circuit_file, circuit_div);
                });
                contents.appendChild(createFigure(circuit_div, "Circuit"));
                const brickwork_div = document.createElement("div");
                downloadJSON(`${brickwork_state_table_dirname}/${json_filename}`, function (table) {
                    brickwork_div.appendChild(createBrickworkStateGraph(table));
                });
                contents.appendChild(createFigure(brickwork_div, "Brickwork state"));
            }
            else {
                keyCell.removeChild(contents);
                contents = null;
            }
        });
        row.appendChild(keyCell);
        const probCell = document.createElement('td');
        probCell.textContent = parseFloat(prob).toFixed(2); // Format as float
        row.appendChild(probCell);
        tbody.appendChild(row);
    });
}

// Function to sort tableData by the given column (0: key, 1: value)
function sortTableByColumn(columnIndex) {
    // Toggle sort direction if same column; otherwise, default to ascending.
    if (currentSortColumn === columnIndex) {
        currentSortDirection *= -1;
    } else {
        currentSortColumn = columnIndex;
        currentSortDirection = 1;
    }
    
    tableData.sort((a, b) => {
        // Sort by key (string) or value (number)
        if (columnIndex === 0) {
            return a[0].localeCompare(b[0]) * currentSortDirection;
        } else {
            return (parseFloat(a[1]) - parseFloat(b[1])) * currentSortDirection;
        }
    });
    renderTable();
}

// Add click event listeners to header cells for sorting.
document.getElementById('circuit-header').addEventListener('click', function() {
    sortTableByColumn(0);
});
document.getElementById('probability-header').addEventListener('click', function() {
    sortTableByColumn(1);
});

function download(url, k) {
    const xhr = new XMLHttpRequest();
    xhr.open("GET", url, true);
    xhr.onreadystatechange = function() {
        if (xhr.readyState === XMLHttpRequest.DONE) {
            if (xhr.status === 200) {
                k(xhr.responseText)
            } else {
                console.error(`Failed to load ${url}. Status:`, xhr.status);
            }
        }
    };
    xhr.send();
}

function downloadJSON(url, k) {
    download(url, function(text) {
        let json;
        try {
          json = JSON.parse(text);
        } catch (e) {
          console.error("Error parsing JSON:", e);
        }
        k(json)
    });
}

export default function renderJSONTable(circuits_dirname, brickwork_state_table_dirname, path_json_table) {
    downloadJSON(path_json_table, function(json) {
        tableData = Object.entries(json);
        renderTable(circuits_dirname, brickwork_state_table_dirname);
    });
}
