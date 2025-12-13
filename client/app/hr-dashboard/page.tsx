"use client";

import React, { useState, useEffect, useRef } from "react";
import { useRouter } from "next/navigation";
import { HRBarChart } from "@/components/hr/BarChart";
import { HRPieChart } from "@/components/hr/PieChart";
import { Sidebar } from "@/components/hr/Sidebar";
import { EmployeeReports } from "@/components/hr/EmployeeReports";
import {
  BarChart as BarChartIcon,
  FileText,
  Search,
  RefreshCw,
} from "lucide-react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { useAuth } from "@/context/AuthContext";
import axios from "axios";
import { UploadData } from "@/components/hr/UploadData";
import { toast } from "sonner"; // Import toast from sonner

interface Employee {
  Employee_ID: string;
  Employee_Name: string;
  Employee_Email: string;
  Employee_Role: string;
  Is_Selected: boolean;
  Sentimental_Score: number;
  Is_Flagged: boolean;
  Report: string;
  Feature_Vector: string;
  Conversation_Completed: boolean;
}

const HRDashboard: React.FC = () => {
  const server = "http://127.0.0.1:8000";

  const [sidebarCollapsed, setSidebarCollapsed] = useState<boolean>(true);
  const [isLoaded, setIsLoaded] = useState<boolean>(false);
  const [searchQuery, setSearchQuery] = useState<string>("");

  const [employees, setEmployees] = useState<Employee[]>([]);
  const [selectedEmployees, setSelectedEmployees] = useState<Employee[]>([]);
  const [employeesWithReports, setEmployeesWithReports] = useState<Employee[]>(
    []
  );

  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [dataLoadingStatus, setDataLoadingStatus] = useState({
    employees: false,
    reports: false,
    generating: false,
  });
  const analyticsRef = useRef<HTMLDivElement | null>(null);
  const reportsRef = useRef<HTMLDivElement | null>(null);

  const router = useRouter();
  const { fetchHRProfile, hrData } = useAuth();

  // useEffect(() => {
  //   const checkAuth = async () => {
  //     setLoading(true);
  //     try {
  //       const profile = await fetchHRProfile();
  //       if (!profile) {
  //         toast.error("Authentication failed", {
  //           description: "You are not authorized to access the HR dashboard",
  //           duration: 5000,
  //           id: "hr-auth-error",
  //         });
  //         localStorage.removeItem("access_token");
  //         router.push("/");
  //         return;
  //       }
  //       toast.success("Welcome to HR Dashboard", {
  //         description: `Logged in as HR Manager`,
  //         duration: 3000,
  //         id: "hr-auth-success",
  //       });
  //     } catch (error) {
  //       console.log(error);
  //       toast.error("Session expired", {
  //         description: "Please log in again to continue",
  //         duration: 5000,
  //         id: "hr-session-error",
  //       });
  //       router.push("/");
  //     } finally {
  //       setLoading(false);
  //     }
  //   };

  //   checkAuth();
  // }, []);

  const fetchAllEmployees = async () => {
    if (dataLoadingStatus.employees) return; // Prevent duplicate API calls

    try {
      setLoading(true);
      setDataLoadingStatus((prev) => ({ ...prev, employees: true }));

      // Show loading toast
      toast.loading("Loading employee data...", { id: "employees-loading" });

      const response = await axios.get(`${server}/api/data/employees`);
      setEmployees(response.data.employees);
      const selectedEmployees = response.data.employees.filter(
        (employee: any) => employee.Is_Selected === true
      );
      console.log("Selected Employee data:", selectedEmployees);
      setSelectedEmployees(selectedEmployees);
      setError(null);

      // Dismiss loading toast and show success
      toast.dismiss("employees-loading");
      toast.success("Data loaded", {
        description: `${selectedEmployees.length} employees retrieved successfully`,
        duration: 3000,
        id: "employees-success", // Adding a unique ID to prevent duplicates
      });
    } catch (error) {
      console.error("Error fetching employees:", error);
      setError("Failed to load employee data");

      // Dismiss loading toast and show error
      toast.dismiss("employees-loading");
      toast.error("Failed to load data", {
        description:
          "Could not retrieve employee data. Please try again later.",
        id: "employees-error", // ID prevents duplicate toasts
        duration: 5000,
      });
    } finally {
      setLoading(false);
      setIsLoaded(true);
      setDataLoadingStatus((prev) => ({ ...prev, employees: false }));
    }
  };

  const fetchTodaysConv = async () => {
    console.log("Fetching Todays Conv:");
    if (dataLoadingStatus.reports) return; // Prevent duplicate API calls

    try {
      setLoading(true);
      setDataLoadingStatus((prev) => ({ ...prev, reports: true }));

      // Show loading toast
      toast.loading("Loading today's reports...", { id: "reports-loading" });

      const response = await axios.get(
        `${server}/api/conversation/todays_reports`
      );
      setEmployeesWithReports(response.data);
      setError(null);

      // Dismiss loading toast and show success
      toast.dismiss("reports-loading");
      toast.success("Reports loaded", {
        description: `${response.data.length} conversation reports retrieved`,
        duration: 3000,
        id: "reports-success", // Adding a unique ID to prevent duplicates
      });
    } catch (error) {
      console.error("Error fetching employees with reports:", error);
      setError("Failed to load employee data");

      // Dismiss loading toast and show error
      toast.dismiss("reports-loading");
      toast.error("Failed to load reports", {
        description:
          "Could not retrieve today's conversation reports. Please try again later.",
        id: "reports-error", // ID prevents duplicate toasts
        duration: 5000,
      });
    } finally {
      setLoading(false);
      setIsLoaded(true);
      setDataLoadingStatus((prev) => ({ ...prev, reports: false }));
    }
  };

  useEffect(() => {
    fetchAllEmployees();
    fetchTodaysConv();
  }, []); // Run once on component mount

  // Ensure HR profile is available for sidebar display
  useEffect(() => {
    if (!hrData) {
      fetchHRProfile().catch((err) =>
        console.error("Failed to fetch HR profile for sidebar:", err)
      );
    }
    console.log(hrData);
  }, [fetchHRProfile, hrData]);

  // Function to regenerate reports for a specific employee
  const regenerateReport = async (employeeId: string) => {
    if (dataLoadingStatus.generating) return; // Prevent duplicate API calls

    try {
      setDataLoadingStatus((prev) => ({ ...prev, generating: true }));

      // Show loading toast
      toast.loading(`Generating report for employee...`, {
        id: `report-gen`,
      });

      // API call to regenerate the report
      const response = await axios.post(
        `${server}/api/conversation/regenerate_report`,
        {
          employee_id: employeeId,
        }
      );

      // Dismiss loading toast and show success
      toast.dismiss(`report-gen`);
      toast.success("Report generated", {
        description: `New report for employee has been created`,
        duration: 3000,
        id: "report-gen-success",
      });

      // Refresh the reports data
      fetchTodaysConv();
    } catch (error) {
      console.error("Error regenerating report:", error);

      // Dismiss loading toast and show error
      toast.dismiss(`report-gen`);
      toast.error("Report generation failed", {
        description: "Unable to generate a new report. Please try again.",
        duration: 5000,
        id: "report-gen-err", // ID prevents duplicate toasts
      });
    } finally {
      setDataLoadingStatus((prev) => ({ ...prev, generating: false }));
    }
  };

  // Function to refresh all data
  const refreshAllData = () => {
    toast.loading("Refreshing dashboard data...", { id: "refresh-all" });

    // Set a timeout to simulate loading and prevent UI flashing
    setTimeout(() => {
      Promise.all([fetchAllEmployees(), fetchTodaysConv()])
        .then(() => {
          toast.dismiss("refresh-all");
          toast.success("Data refreshed", {
            description: "All dashboard data has been updated",
            duration: 3000,
            id: "refresh-success", // Adding a unique ID to prevent duplicates
          });
        })
        .catch(() => {
          toast.dismiss("refresh-all");
          toast.error("Refresh failed", {
            description: "Could not update all dashboard data",
            duration: 5000,
            id: "refresh-error", // ID prevents duplicate toasts
          });
        });
    }, 500);
  };

  const selectedStats = employees.reduce(
    (acc, employee) => {Sidebar
      if (employee.Is_Selected) {
        acc.selected++;
      } else {
        acc.notSelected++;
      }
      return acc;
    },
    { selected: 0, notSelected: 0 }
  );

  return (
    <div className="flex h-screen bg-hr-black overflow-hidden ml-15">
      <Sidebar
        collapsed={sidebarCollapsed}
        onToggle={() => setSidebarCollapsed(!sidebarCollapsed)}
        hrName={hrData?.hr_name}
        hrEmail={hrData?.hr_email}
        hrId={hrData?.hr_id}
      />

      <main className="flex-1 transition-all duration-300 ease-in-out overflow-y-auto hr-custom-scrollbar">
        <div className="p-6 space-y-8">
          {/* Header with refresh button */}
          <div className="flex justify-between items-center">
            <h1 className="text-3xl font-bold text-white">HR Dashboard</h1>
            <Button
              onClick={refreshAllData}
              variant="outline"
              size="sm"
              className="flex items-center gap-2 bg-transparent border border-[#26890d]/50 text-[#26890d] hover:bg-[#26890d]/10"
              disabled={
                dataLoadingStatus.employees || dataLoadingStatus.reports
              }
            >
              <RefreshCw
                size={16}
                className={
                  dataLoadingStatus.employees || dataLoadingStatus.reports
                    ? "animate-spin"
                    : ""
                }
              />
              Refresh Data
            </Button>
          </div>

          {/* Analytics Section */}
          <div id="analytics-section" ref={analyticsRef} className="space-y-6">
            <h2 className="text-2xl font-bold text-[#26890d] mb-4 flex items-center gap-2">
              <BarChartIcon size={24} className="text-[#26890d]" />
              ANALYTICS DASHBOARD
            </h2>

            <div
              className={`grid grid-cols-1 lg:grid-cols-2 gap-6 ${
                isLoaded ? "scale-in" : "opacity-0"
              }`}
            >
              {loading ? (
                <div className="col-span-2 flex items-center justify-center h-64">
                  <p className="text-hr-green">Loading data...</p>
                </div>
              ) : error ? (
                <div className="col-span-2 flex items-center justify-center h-64">
                  <p className="text-red-500">{error}</p>
                </div>
              ) : (
                <>
                  <HRPieChart
                    flaggedCount={selectedStats.selected}
                    unflaggedCount={selectedStats.notSelected}
                  />
                  <HRBarChart
                  // employees={employees}
                  />
                </>
              )}
            </div>
          </div>

          {/* Reports Section */}
          <div id="reports-section" ref={reportsRef} className="space-y-6 pt-8">
            <h2 className="text-2xl font-bold text-[#26890d] mb-4 flex items-center gap-2">
              <FileText size={24} className="text-[#26890d]" />
              Employee Reports
            </h2>

            <div
              className={`${
                isLoaded ? "scale-in" : "opacity-0"
              } animation-delay-700`}
            >
              <div className="mb-4 flex items-center">
                <div className="relative w-full max-w-md">
                  <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 h-4 w-4 pointer-events-none" />
                  <Input
                    className="pl-10 bg-hr-black border border-[#26890d]/30 focus-visible:ring-0 focus-visible:ring-offset-0 focus-visible:border-[#26890d]/100 focus-visible:border text-gray-200 placeholder:text-gray-500 outline-none transition-all duration-200"
                    placeholder="Search by employee ID or name..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    type="text"
                    autoComplete="off"
                    spellCheck="false"
                    style={{ boxShadow: "none" }}
                  />
                </div>
              </div>

              {loading ? (
                <div className="flex items-center justify-center h-40">
                  <p className="text-hr-green">Loading reports...</p>
                </div>
              ) : error ? (
                <div className="flex items-center justify-center h-40">
                  <p className="text-red-500">{error}</p>
                </div>
              ) : (
                <EmployeeReports
                  searchQuery={searchQuery}
                  employees={selectedEmployees}
                  employeesWithReports={employeesWithReports}
                  onRegenerateReport={regenerateReport}
                  isGenerating={dataLoadingStatus.generating}
                />
              )}
            </div>
          </div>

          {/* Upload Data Section */}
          <div id="upload-data-section" className="space-y-6 pt-8">
            <h2 className="text-2xl font-bold text-[#26890d] mb-4 flex items-center gap-2">
              <FileText size={24} className="text-[#26890d]" />
              Upload Data
            </h2>
            <UploadData />
          </div>
        </div>
      </main>
    </div>
  );
};

export default HRDashboard;
