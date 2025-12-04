import { NextResponse } from "next/server";
import fs from "fs";
import path from "path";
import fetch from "node-fetch";

interface TokenResponse {
  access_token: string;
}

// Generate next filename like true-color-1.png, true-color-2.png, ...
async function getNextFilename() {
  const dir = path.join(process.cwd(), "public", "sentinel");

  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }

  const files = fs.readdirSync(dir)
    .filter(f => f.startsWith("true-color-") && f.endsWith(".png"));

  const numbers = files
    .map(f => parseInt(f.replace("true-color-", "").replace(".png", "")))
    .filter(n => !isNaN(n));

  const next = numbers.length > 0 ? Math.max(...numbers) + 1 : 1;

  return `true-color-${next}.png`;
}

async function downloadImage(imageBase64: string, filename: string) {
  const buffer = Buffer.from(imageBase64.split(",")[1], "base64");

  const filePath = path.join(process.cwd(), "public", "sentinel", filename);

  if (!fs.existsSync(path.dirname(filePath))) {
    fs.mkdirSync(path.dirname(filePath), { recursive: true });
  }

  fs.writeFileSync(filePath, buffer);

  return `/sentinel/${filename}`;
}

export async function POST(req: Request) {
  try {
    const { lat1, lon1, lat2, lon2 } = await req.json();

    const minLat = Math.min(lat1, lat2);
    const maxLat = Math.max(lat1, lat2);
    const minLon = Math.min(lon1, lon2);
    const maxLon = Math.max(lon1, lon2);
    const bbox = [minLon, minLat, maxLon, maxLat];

    const CLIENT_ID = process.env.SENTINELHUB_CLIENT_ID!;
    const CLIENT_SECRET = process.env.SENTINELHUB_CLIENT_SECRET!;

    // Authenticate
    const tokenRes = await fetch("https://services.sentinel-hub.com/oauth/token", {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: new URLSearchParams({
        grant_type: "client_credentials",
        client_id: CLIENT_ID,
        client_secret: CLIENT_SECRET,
      }),
    });

    const tokenData = (await tokenRes.json()) as TokenResponse;
    const access_token = tokenData.access_token;

    if (!access_token) throw new Error("Failed to authenticate with SentinelHub");

    // True Color Script
    const evalscriptTrueColor = `
      //VERSION=3
      function setup() {
        return { input: ["B04","B03","B02"], output: { bands: 3 } };
      }
      function evaluatePixel(s) {
        return [s.B04 * 2.5, s.B03 * 2.5, s.B02 * 2.5];
      }
    `;

    async function fetchImage(evalscript: string) {
      const body = {
        input: {
          bounds: {
            bbox,
            properties: { crs: "http://www.opengis.net/def/crs/EPSG/0/4326" },
          },
          data: [
            {
              type: "sentinel-2-l2a",
              dataFilter: {
                timeRange: {
                  from: new Date(new Date().setMonth(new Date().getMonth() - 2)).toISOString(),
                  to: new Date().toISOString(),
                },
                maxCloudCoverage: 20,
              },
            },
          ],
        },
        output: { width: 512, height: 512, responses: [{ identifier: "default", format: { type: "image/png" } }] },
        evalscript,
      };

      const res = await fetch("https://services.sentinel-hub.com/api/v1/process", {
        method: "POST",
        headers: { Authorization: `Bearer ${access_token}`, "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });

      if (!res.ok) throw new Error(await res.text());

      const buf = await res.arrayBuffer();
      return `data:image/png;base64,${Buffer.from(buf).toString("base64")}`;
    }

    const trueColorBase64 = await fetchImage(evalscriptTrueColor);

    // USE AUTO-INCREMENTING FILENAME
    const filename = await getNextFilename();

    const fileUrl = await downloadImage(trueColorBase64, filename);

    return NextResponse.json({
      trueColorUrl: fileUrl,
      filename,
      bbox,
    });

  } catch (err: any) {
    console.error(err);
    return NextResponse.json({ error: err.message }, { status: 500 });
  }
}
