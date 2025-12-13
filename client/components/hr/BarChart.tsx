import React, { useEffect, useState } from "react";
import {
  BarChart as RechartsBC,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

// Updated data to use abbreviated day names
const data = [
  { day: "Mon", Excited: 22, Happy: 35, Okay: 18, Sad: 15, Frustrated: 10 },
  { day: "Tue", Excited: 25, Happy: 38, Okay: 15, Sad: 12, Frustrated: 8 },
  { day: "Wed", Excited: 18, Happy: 30, Okay: 22, Sad: 18, Frustrated: 12 },
  { day: "Thu", Excited: 28, Happy: 40, Okay: 16, Sad: 10, Frustrated: 6 },
  { day: "Fri", Excited: 24, Happy: 36, Okay: 20, Sad: 12, Frustrated: 8 },
];

export function HRBarChart() {
  const [hoveredEmotion, setHoveredEmotion] = useState<string | null>(null);
  const [hoveredDay, setHoveredDay] = useState<string | null>(null);

  const emotions = ["Excited", "Happy", "Okay", "Sad", "Frustrated"];
  const colors: { [key: string]: string } = {
    Excited: "#26890d", // dark green with more opacity
    Happy: "#CEFF00", // bright yellow-green, more different from dark green
    Okay: "#facc15", // yellow
    Sad: "#3b82f6", // blue
    Frustrated: "#ef4444", // red
  };

  const handleMouseEnter = (emotion: string) => {
    setHoveredEmotion(emotion);
  };

  const handleMouseLeave = () => {
    setHoveredEmotion(null);
  };

  const getBarOpacity = (emotion: string) => {
    if (!hoveredEmotion) return 1;
    return emotion === hoveredEmotion ? 1 : 0.5;
  };

  const renderCustomizedLegend = (props: any) => {
    const { payload } = props;

    return (
      <div className="flex flex-wrap justify-center gap-4 mt-3">
        {payload.map((entry: any, index: number) => (
          <div
            key={`item-${index}`}
            className={`flex items-center cursor-pointer transition-opacity ${
              !hoveredEmotion || entry.value === hoveredEmotion ? "opacity-100" : "opacity-50"
            }`}
            onMouseEnter={() => handleMouseEnter(entry.value)}
            onMouseLeave={handleMouseLeave}
          >
            <div
              className="chart-legend-item"
              style={{
                backgroundColor: entry.color,
              }}
            />
            <span className="text-sm">{entry.value}</span>
          </div>
        ))}
      </div>
    );
  };

  return (
    <Card className="shadow-card bg-card border border-[#26890d]/20">
      <CardHeader>
        <CardTitle className="text-xl font-semibold text-white">
          Employee Mood Trends
        </CardTitle>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={350}>
          <RechartsBC
            data={data}
            margin={{
              top: 20,
              right: 30,
              left: 20,
              bottom: 5,
            }}
            onMouseEnter={(data) => {
              if (data && data.activePayload && data.activePayload[0]) {
                setHoveredDay(data.activeLabel || null);
              }
            }}
            onMouseLeave={() => {
              setHoveredDay(null);
            }}
          >
            <CartesianGrid
              strokeDasharray="3 3"
              stroke="rgba(255,255,255,0.1)"
            />
            <XAxis
              dataKey="day"
              stroke="rgba(255,255,255,0.5)"
              tick={{ fontSize: "0.8rem" }}
              interval={0}
              height={50}
              tickMargin={8}
              className="chart-axis-tick"
            />
            <YAxis
              stroke="rgba(255,255,255,0.5)"
              tick={{ fontSize: "0.8rem" }}
              width={40}
              className="chart-axis-tick"
            />
            <Tooltip
              contentStyle={{
                backgroundColor: "#1a1a1a",
                borderColor: "rgba(74, 222, 128, 0.3)",
                color: "white",
              }}
              cursor={{
                fill: "rgba(38, 137, 13, 0.3)",
                stroke: "rgba(38, 137, 13, 0.7)",
                strokeWidth: 1,
              }}
            />
            <Legend content={renderCustomizedLegend} />
            {emotions.map((emotion) => (
              <Bar
                key={emotion}
                dataKey={emotion}
                fill={colors[emotion]}
                onMouseEnter={() => handleMouseEnter(emotion)}
                onMouseLeave={handleMouseLeave}
                opacity={getBarOpacity(emotion)}
                strokeWidth={0}
                stroke="none"
                activeBar={{
                  strokeWidth: 0,
                  stroke: "transparent",
                  fill: colors[emotion],
                }}
                className="chart-bar-no-highlight"
              />
            ))}
          </RechartsBC>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}
