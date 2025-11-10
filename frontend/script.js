document.getElementById("submitBtn").addEventListener("click", async () => {
  const query = document.getElementById("queryInput").value.trim();
  const resultsDiv = document.getElementById("results");
  resultsDiv.innerHTML = "<p>Loading recommendations...</p>";

  if (!query) {
    resultsDiv.innerHTML = "<p style='color:red;'>Please enter a query.</p>";
    return;
  }

  try {
    const response = await fetch(
      `https://product-recommendation-x7fp.onrender.com/recommend`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          query: query,
          input_type: "text"
        })
      }
    );

    if (!response.ok) throw new Error("API request failed");

    const data = await response.json();
    const recs = data.recommended_assessments || [];

    if (recs.length === 0) {
      resultsDiv.innerHTML = "<p>No recommendations found.</p>";
      return;
    }

    resultsDiv.innerHTML = recs.map(rec => {
      // Ensure duration is always a number or "N/A"
      const duration = rec.duration != null ? rec.duration : "N/A";

      // Ensure test_type is always an array
      const testType = Array.isArray(rec.test_type) ? rec.test_type.join(", ") : rec.test_type;

      return `
        <div class="card">
          <h3><a href="${rec.url}" target="_blank">${rec.name}</a></h3>
          <p><b>Description:</b> ${rec.description}</p>
          <p><b>Duration:</b> ${duration}</p>
          <p><b>Remote Support:</b> ${rec.remote_support}</p>
          <p><b>Adaptive Support:</b> ${rec.adaptive_support}</p>
          <p><b>Test Type:</b> ${testType}</p>
        </div>
      `;
    }).join("");

  } catch (error) {
    console.error(error);
    resultsDiv.innerHTML = "<p style='color:red;'>Error fetching recommendations. Please try again.</p>";
  }
});
