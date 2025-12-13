import React, { useEffect, useState, useCallback } from "react";
import {
  PieChart as RechartsPC,
  Pie,
  Cell,
  ResponsiveContainer,
  Legend,
  Tooltip,
  PieLabelRenderProps,
} from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

interface HRPieChartProps {
  flaggedCount: number;
  unflaggedCount: number;
}

const COLORS = ["#86BC25", "#E11D48"] as const;

export function HRPieChart({ flaggedCount, unflaggedCount }: HRPieChartProps) {
  const [isAnimating, setIsAnimating] = useState(false);
  const [activeSlice, setActiveSlice] = useState<number | null>(null);
  const [hoverSlice, setHoverSlice] = useState<number | null>(null);
  const [windowWidth, setWindowWidth] = useState(0);

  useEffect(() => {
    setIsAnimating(true);
    setWindowWidth(window.innerWidth);

    const handleResize = () => setWindowWidth(window.innerWidth);
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  const data = [
    { name: "Unflagged", value: unflaggedCount, color: COLORS[0] },
    { name: "Flagged", value: flaggedCount, color: COLORS[1] },
  ];

  const renderColorfulLegendText = (value: string, entry: any) => {
    const color = entry.color;
    return (
      <span className="hr-chart-legend-text" style={{ color }}>
        {value}
      </span>
    );
  };

  const handleClick = (_: any, index: number) => {
    setActiveSlice(activeSlice === index ? null : index);
  };

  const renderCustomizedLabel = useCallback(
    (props: PieLabelRenderProps) => {
      const {
        cx = 0,
        cy = 0,
        midAngle = 0,
        innerRadius = 0,
        outerRadius = 0,
        percent = 0,
        name = "",
        index = 0,
      } = props;

      if (windowWidth < 380) return null;

      // Convert all values to numbers explicitly
      const numCx = Number(cx);
      const numCy = Number(cy);
      const numMidAngle = Number(midAngle);
      const numInnerRadius = Number(innerRadius);
      const numOuterRadius = Number(outerRadius);
      const numPercent = Number(percent);
      const numIndex = Number(index);

      const RADIAN = Math.PI / 180;
      const isLargeScreen = windowWidth >= 768;

      if (isLargeScreen) {
        const radius = numOuterRadius * 1.1;
        const x = numCx + radius * Math.cos(-numMidAngle * RADIAN);
        const y = numCy + radius * Math.sin(-numMidAngle * RADIAN);

        // Create connecting line
        const lineX1 =
          numCx + numOuterRadius * 0.95 * Math.cos(-numMidAngle * RADIAN);
        const lineY1 =
          numCy + numOuterRadius * 0.95 * Math.sin(-numMidAngle * RADIAN);
        const lineX2 =
          numCx + numOuterRadius * 1.05 * Math.cos(-numMidAngle * RADIAN);
        const lineY2 =
          numCy + numOuterRadius * 1.05 * Math.sin(-numMidAngle * RADIAN);

        return (
          <>
            <line
              x1={lineX1}
              y1={lineY1}
              x2={lineX2}
              y2={lineY2}
              stroke={COLORS[numIndex % COLORS.length]}
              className="hr-chart-line"
            />
            <text
              x={x}
              y={y}
              fill={COLORS[numIndex % COLORS.length]}
              textAnchor={x > numCx ? "start" : "end"}
              dominantBaseline="central"
              fontSize="13px"
              fontWeight="600"
              style={{
                filter: "drop-shadow(0px 0px 1px rgba(0, 0, 0, 0.8))",
                pointerEvents: "none",
              }}
            >
              {`${name}: ${(numPercent * 100).toFixed(0)}%`}
            </text>
          </>
        );
      } else {
        const radius = numInnerRadius + (numOuterRadius - numInnerRadius) * 0.6;
        const x = numCx + radius * Math.cos(-numMidAngle * RADIAN);
        const y = numCy + radius * Math.sin(-numMidAngle * RADIAN);

        return (
          <text
            x={x}
            y={y}
            fill="#ffffff"
            textAnchor={x > numCx ? "start" : "end"}
            dominantBaseline="central"
            fontSize="12px"
            fontWeight="500"
            style={{ pointerEvents: "none" }}
          >
            {`${name} ${(numPercent * 100).toFixed(0)}%`}
          </text>
        );
      }
    },
    [windowWidth]
  );

  return (
    <Card className={`hr-chart-card ${isAnimating ? "hr-chart-appear" : ""}`}>
      <CardHeader className="pb-2">
        <CardTitle className="hr-chart-title">Employee Flag Status</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="hr-chart-container">
          <ResponsiveContainer width="100%" height="100%">
            <RechartsPC>
              <Pie
                data={data}
                cx="50%"
                cy="50%"
                labelLine={false}
                outerRadius={
                  windowWidth < 576 ? 80 : windowWidth >= 768 ? 90 : 100
                }
                dataKey="value"
                animationDuration={800}
                label={renderCustomizedLabel}
                onClick={handleClick}
                onMouseEnter={(data, index) => {
                  setHoverSlice(index);
                }}
                onMouseLeave={() => {
                  setHoverSlice(null);
                }}
                isAnimationActive={false}
                strokeWidth={1}
                stroke="#131313"
              >
                {data.map((_, index) => (
                  <Cell
                    key={`cell-${index}`}
                    fill={COLORS[index % COLORS.length]}
                    strokeWidth={
                      activeSlice === index || hoverSlice === index ? 3 : 1
                    }
                    stroke={
                      activeSlice === index || hoverSlice === index
                        ? COLORS[index % COLORS.length]
                        : "#131313"
                    }
                  />
                ))}
              </Pie>
              <Tooltip
                formatter={(value: number) => [`${value} employees`]}
                contentStyle={{
                  backgroundColor: "rgba(20, 20, 20, 0.95)",
                  borderRadius: "8px",
                  border: "none",
                  color: "#f0f0f0",
                }}
                labelStyle={{
                  color: "#ffffff"
                }}
                itemStyle={{
                  color: "#ffffff"
                }}
              />
              <Legend
                layout="horizontal"
                verticalAlign="bottom"
                formatter={renderColorfulLegendText}
              />
            </RechartsPC>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
}
