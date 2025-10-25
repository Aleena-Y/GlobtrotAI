import { useEffect, useState, useRef } from "react";
import { MapContainer, TileLayer, GeoJSON } from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";

// Public GeoJSON source
const GEOJSON_URL =
  "https://raw.githubusercontent.com/datasets/geo-countries/master/data/countries.geojson";

export default function App() {
  const [recommendations, setRecommendations] = useState([]);
  const [geoData, setGeoData] = useState(null);
  const [searchInput, setSearchInput] = useState("");
  const [highlightCountries, setHighlightCountries] = useState([]);
  const geoJsonRef = useRef(null);
  const mapRef = useRef(null);
  const [rationale, setRationale] = useState({});
  const [tips, setTips] = useState({});
  const [isLoading, setIsLoading] = useState(false);

  // Load GeoJSON
  useEffect(() => {
    fetch(GEOJSON_URL)
      .then((r) => r.json())
      .then((data) => setGeoData(data))
      .catch((err) => console.error("Failed to load geojson:", err));
  }, []);

  // Styles
  const defaultStyle = {
    fillColor: "transparent",
    fillOpacity: 0,
    color: "#222",
    weight: 1,
    interactive: true,
  };
  const dimStyle = {
    fillColor: "gray",
    fillOpacity: 0.4,
    color: "#222",
    weight: 1,
    interactive: true,
  };
  const highlightStyle = {
    fillColor: "yellow",
    fillOpacity: 0.7,
    color: "red",
    weight: 3,
    interactive: true,
  };

  // Normalize country names
  const normalize = (name) => (name ? name.trim().toLowerCase() : "");

  // Fetch recommendations
  const fetchRecommendations = async () => {
    if (!searchInput.trim()) return;

    setIsLoading(true);
    try {
      const url = new URL("http://localhost:8000/recommend");
      url.searchParams.append("query", searchInput.trim());

      const response = await fetch(url);
      if (!response.ok) throw new Error(`HTTP error ${response.status}`);
      const res = await response.json();

      console.log("API Response:", res);

      // Extract recommendations
      let finalList = [];
      if (Array.isArray(res.recommendations) && res.recommendations.length > 0) {
        finalList = res.recommendations.map((c) => String(c).trim());
      } else if (Array.isArray(res.retrieved)) {
        finalList = res.retrieved
          .map((r) => {
            if (typeof r === "string") return r.trim();
            if (r?.text) {
              const match = r.text.match(/^([^(]+)\s*\(/);
              return match ? match[1].trim() : "";
            }
            return "";
          })
          .filter(Boolean);
      }

      // Store normalized data
      setRecommendations(finalList);
      setHighlightCountries(finalList.map((c) => normalize(c)));

      const normalizedRationale = {};
      Object.entries(res.rationale || {}).forEach(([k, v]) => {
        normalizedRationale[normalize(k)] = v;
      });
      setRationale(normalizedRationale);

      const normalizedTips = {};
      Object.entries(res.tips || {}).forEach(([k, v]) => {
        normalizedTips[normalize(k)] = v;
      });
      setTips(normalizedTips);

      console.log("Final recommendations:", finalList);
    } catch (err) {
      console.error("API error:", err);
      alert("Error fetching recommendations. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  // GeoJSON handlers
  const onEachFeature = (feature, layer) => {
    const name = feature?.properties?.ADMIN || feature?.properties?.name;
    if (!name) return;

    layer.bindPopup(name);

    layer.on("click", () => {
      const key = normalize(name);
      const rationaleText = rationale[key];
      const tipText = tips[key];

      let message = `Country: ${name}\n\n`;
      if (rationaleText) message += `Why recommended: ${rationaleText}\n\n`;
      if (tipText) message += `Travel tip: ${tipText}`;
      if (!rationaleText && !tipText) {
        message += "No additional information available for this country.";
      }
      alert(message);
    });
  };

  const pointToLayer = (_feature, latlng) =>
    L.circleMarker(latlng, { radius: 3, weight: 0.5 });

  // Search handling
  const handleSearch = () => fetchRecommendations();
  const handleKeyPress = (e) => e.key === "Enter" && handleSearch();
  const handleClear = () => {
    setSearchInput("");
    setHighlightCountries([]);
    setRecommendations([]);
    setRationale({});
    setTips({});
  };

  // Styling
  const getFeatureStyle = (feature) => {
    const name = feature?.properties?.ADMIN || feature?.properties?.name;
    if (!name) return defaultStyle;
    if (highlightCountries.length === 0) return defaultStyle;

    const isMatch = highlightCountries.includes(normalize(name));
    return isMatch ? highlightStyle : dimStyle;
  };

  // Count matches
  const getMatchCount = () => {
    if (!geoData || highlightCountries.length === 0) return 0;
    return geoData.features.filter((f) =>
      highlightCountries.includes(normalize(f?.properties?.ADMIN || f?.properties?.name))
    ).length;
  };

  // Auto zoom to highlights
  useEffect(() => {
    if (!geoJsonRef.current || !mapRef.current) return;

    let matchedBounds = null;

    geoJsonRef.current.eachLayer((subLayer) => {
      const name =
        subLayer?.feature?.properties?.ADMIN ||
        subLayer?.feature?.properties?.name;
      if (!name) return;

      const isMatch = highlightCountries.includes(normalize(name));
      if (subLayer.setStyle) {
        subLayer.setStyle(isMatch ? highlightStyle : highlightCountries.length > 0 ? dimStyle : defaultStyle);
      }

      if (isMatch && subLayer.getBounds) {
        matchedBounds = matchedBounds
          ? matchedBounds.extend(subLayer.getBounds())
          : subLayer.getBounds();
      }
    });

    if (matchedBounds) {
      try {
        mapRef.current.fitBounds(matchedBounds, { maxZoom: 6, padding: [20, 20] });
      } catch (e) {
        console.warn("fitBounds error:", e);
      }
    }
  }, [highlightCountries, geoData]);

  return (
    <div style={{ height: "100vh", display: "flex", flexDirection: "column" }}>
      {/* Top search bar */}
      <div style={{ padding: 12, backgroundColor: "#f5f5f5", borderBottom: "1px solid #ddd" }}>
        <div style={{ display: "flex", gap: 8, alignItems: "center", maxWidth: 900 }}>
          <input
            type="text"
            placeholder="Type a query — e.g. affordable Asian country with good purchasing power"
            value={searchInput}
            onChange={(e) => setSearchInput(e.target.value)}
            onKeyPress={handleKeyPress}
            disabled={isLoading}
            style={{
              flex: 1,
              padding: 8,
              border: "1px solid #ccc",
              borderRadius: 4,
              fontSize: 14,
              opacity: isLoading ? 0.7 : 1,
            }}
          />
          <button
            onClick={handleSearch}
            disabled={isLoading || !searchInput.trim()}
            style={{
              padding: "8px 16px",
              backgroundColor: isLoading ? "#6c757d" : "#007bff",
              color: "white",
              border: "none",
              borderRadius: 4,
              cursor: isLoading ? "not-allowed" : "pointer",
              fontSize: 14,
              fontWeight: "500",
            }}
          >
            {isLoading ? "Searching..." : "Search"}
          </button>
          <button
            onClick={handleClear}
            disabled={isLoading}
            style={{
              padding: "8px 16px",
              backgroundColor: "#6c757d",
              color: "white",
              border: "none",
              borderRadius: 4,
              cursor: isLoading ? "not-allowed" : "pointer",
              fontSize: 14,
              fontWeight: "500",
            }}
          >
            Clear
          </button>
        </div>
        {highlightCountries.length > 0 && (
          <div style={{ marginTop: 8, fontSize: 12, color: "#666" }}>
            Recommended countries:{" "}
            {recommendations.map((c, i) => (
              <span key={i}>
                <strong>{c}</strong>
                {i < recommendations.length - 1 && ", "}
              </span>
            ))}
            {geoData && ` (${getMatchCount()} matches found on map)`}
          </div>
        )}
      </div>

      {/* Map */}
      <div style={{ flex: 1, position: "relative" }}>
        <MapContainer ref={mapRef} center={[20, 0]} zoom={2} style={{ height: "100%", width: "100%" }}>
          <TileLayer
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            attribution="&copy; OpenStreetMap contributors"
          />

          {geoData && (
            <GeoJSON
              key={highlightCountries.join(",")}
              data={geoData}
              ref={geoJsonRef}
              style={getFeatureStyle}
              onEachFeature={onEachFeature}
              pointToLayer={pointToLayer}
            />
          )}
        </MapContainer>

        {/* Recommendations panel */}
        {recommendations.length > 0 && (
          <div
            style={{
              position: "absolute",
              bottom: 20,
              left: 20,
              background: "white",
              padding: 16,
              border: "1px solid #ccc",
              borderRadius: 8,
              boxShadow: "0 4px 12px rgba(0,0,0,0.15)",
              zIndex: 1000,
              fontSize: 14,
              maxWidth: 350,
              maxHeight: "60vh",
              overflowY: "auto",
            }}
          >
            <div
              style={{
                fontSize: 16,
                fontWeight: "bold",
                marginBottom: 12,
                color: "#333",
                borderBottom: "2px solid #007bff",
                paddingBottom: 8,
              }}
            >
              🌍 AI Recommended Countries
            </div>

            <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
              {recommendations.map((country, i) => {
                const key = normalize(country);
                const rationaleText = rationale[key];
                const tipText = tips[key];

                return (
                  <div
                    key={i}
                    style={{
                      padding: 14,
                      borderRadius: 8,
                      background: "#f8f9fa",
                      boxShadow: "0 2px 4px rgba(0,0,0,0.08)",
                      border: "1px solid #e9ecef",
                      cursor: "pointer",
                      transition: "all 0.2s ease",
                    }}
                    onClick={() => {
                      let message = `${country}\n\n`;
                      if (rationaleText) message += `Why recommended: ${rationaleText}\n\n`;
                      if (tipText) message += `💡 Travel tip: ${tipText}`;
                      if (!rationaleText && !tipText)
                        message += "Click on the country on the map for more details.";
                      alert(message);
                    }}
                  >
                    <div style={{ fontSize: 16, fontWeight: 600, color: "#495057", marginBottom: rationaleText || tipText ? 8 : 0 }}>
                      {i + 1}. {country}
                    </div>

                    {rationaleText && (
                      <div style={{ fontSize: 13, color: "#6c757d", marginBottom: tipText ? 8 : 0, lineHeight: "1.4" }}>
                        <span style={{ fontWeight: 500 }}>Why:</span> {rationaleText}
                      </div>
                    )}

                    {tipText && (
                      <div style={{ fontSize: 13, color: "#28a745", fontStyle: "italic", lineHeight: "1.4" }}>
                        💡 {tipText}
                      </div>
                    )}

                    {!rationaleText && !tipText && (
                      <div style={{ fontSize: 12, color: "#adb5bd", fontStyle: "italic" }}>
                        Click for more details
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
