// function buildMetadata(sample) {
//   var url = "/metadata/" + sample;
//   d3.json(url).then(function (response) {
//     console.log(response);
//     d3.select("#metadata")
//       .selectAll("div").remove();
//     d3.select("#metadata")
//       .selectAll("div")
//       .data(Object.entries(response))
//       .enter()
//       .append("div")
//       .text(function (d) {
//         return d[0] + ': ' + d[1];
//       });
//   });
// }



// var submit = d3.select("#submit1");

// submit.on("click", function () {
//     // Prevent the page from refreshing
//     d3.event.preventDefault();

//     // Select the input element and get the raw HTML node
//     var inputValue1 = d3.select("#input1").property("value");
//     var inputValue2 = d3.select("#input2").property("value");
//     var inputValue3 = d3.select("#input3").property("value");

//     var names = [inputValue1, inputValue2, inputValue3];
//     plot(names);
//     console.log(names);
// });



var trace1 = {
    labels: ["beer", "wine", "martini", "margarita",
       "ice tea", "rum & coke", "mai tai", "gin & tonic"],
    values: [22.7, 17.1, 9.9, 8.7, 7.2, 6.1, 6.0, 4.6],
    type: "pie"
 };
var layout = {
    title: "Model Factors Weight Chart"
};
     
var data = [trace1];
Plotly.newPlot("pieChart", data, layout);
console.log("Plot Pie Chart");




// // Initialize the dashboard
// init();
