import React, { useEffect, useMemo, useState } from "react";
// import { Link } from "react-router-dom";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { FileText, Download, View } from "lucide-react";
import { report } from "process";

interface EmployeeReport {
  id: string;
  name: string;
  position: string;
  flagged: boolean;
  conversation_completed: boolean;
  report_link: string;
}
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

interface EmployeeReportsProps {
  searchQuery?: string;
  employees: Employee[];
  employeesWithReports: Employee[];
  onRegenerateReport: (employeeId: string) => void;
  isGenerating: boolean;
}

export function EmployeeReports({
  searchQuery = "",
  employees = [],
  employeesWithReports = [],
}: EmployeeReportsProps) {
  // console.log("Props received:", employees);
  const [hoveredRow, setHoveredRow] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState("all");

  const selectedEmployee: EmployeeReport[] = useMemo(
    () =>
      (employees || []).map((employee) => ({
        id: employee.Employee_ID,
        name: employee.Employee_Name,
        position: employee.Employee_Role,
        flagged: employee.Is_Flagged,
        conversation_completed: employee.Conversation_Completed,
        report_link: "#",
      })),
    [employees]
  );

  const conversed_employees: EmployeeReport[] = useMemo(
    () =>
      (employeesWithReports || []).map((employee) => ({
        id: employee.Employee_ID,
        name: employee.Employee_Name,
        position: employee.Employee_Role,
        flagged: employee.Is_Flagged,
        conversation_completed: employee.Conversation_Completed,
        report_link: employee.Report,
      })),
    [employeesWithReports]
  );

  // useEffect(() => {
  //   if (employees) {
  //     console.log("Employee data received:", selectedEmployee);
  //   }
  // }, [employees]);

  // useEffect(() => {
  //   if (filteredReports) {
  //     console.log("Filtered Reports:", filteredReports);
  //   }
  // }, [activeTab]);

  const handleDownload = (pdfUrl: string) => {
    window.open(pdfUrl, "_blank");
  };

  // Filter reports based on search query
  const filterSelectedEmp = searchQuery
    ? selectedEmployee.filter(
        (report) =>
          report.id.toLowerCase().includes(searchQuery.toLowerCase()) ||
          report.name.toLowerCase().includes(searchQuery.toLowerCase())
      )
    : selectedEmployee;

  const filterConversedEmp = searchQuery
    ? conversed_employees.filter(
        (report) =>
          report.id.toLowerCase().includes(searchQuery.toLowerCase()) ||
          report.name.toLowerCase().includes(searchQuery.toLowerCase())
      )
    : conversed_employees;

  // Filter the reports based on the active tab
  // const filteredReports =
  //   activeTab === "all"
  //     ? filterSelectedEmp
  //     : filterSelectedEmp.filter((report) =>
  //         activeTab === "flagged" ? report.flagged :report.flagged
  //       );

  useEffect(() => {
    console.log("Conversed Employees:", conversed_employees);
  }, [conversed_employees]);

  const filteredReports = useMemo(() => {
    if (activeTab === "all") return filterSelectedEmp;
    if (activeTab === "flagged")
      return filterConversedEmp.filter((r) => r.flagged);
    if (activeTab === "unflagged")
      return filterConversedEmp.filter((r) => !r.flagged);
    return [];
  }, [activeTab, filterSelectedEmp, filterConversedEmp]);

  return (
    <Card className="shadow-card bg-card border border-[#26890d]/30">
      <CardHeader className="pb-2">
        <CardTitle className="text-xl font-medium text-[#26890d] flex items-center gap-2">
          <FileText size={20} className="text-[#26890d]" />
          Employee Reports
        </CardTitle>
      </CardHeader>
      <CardContent>
        <Tabs
          defaultValue="all"
          className="w-full"
          onValueChange={setActiveTab}
        >
          <TabsList className="w-full mb-4 grid grid-cols-3 gap-1 sm:gap-2 md:gap-3 bg-transparent border-0 shadow-none p-0">
            <TabsTrigger
              value="all"
              className="data-[state=active]:bg-[#26890d] data-[state=active]:!text-black text-white hover:bg-[#0f3d17] transition-colors duration-200 border border-[#26890d]/60 rounded-md px-2 sm:px-3 md:px-4 py-1 md:py-2 text-xs sm:text-sm md:text-base flex justify-center items-center whitespace-nowrap"
            >
              <span className="sm:hidden data-[state=active]:!text-black">
                All
              </span>
              <span className="hidden sm:block data-[state=active]:!text-black">
                All Reports
              </span>
            </TabsTrigger>
            <TabsTrigger
              value="unflagged"
              className="data-[state=active]:bg-[#26890d] data-[state=active]:!text-black text-white hover:bg-[#0f3d17] transition-colors duration-200 border border-[#26890d]/60 rounded-md px-2 sm:px-3 md:px-4 py-1 md:py-2 text-xs sm:text-sm md:text-base flex justify-center items-center whitespace-nowrap"
            >
              <span className="sm:hidden data-[state=active]:!text-black">
                Unflagged
              </span>
              <span className="hidden sm:block data-[state=active]:!text-black">
                Unflagged Reports
              </span>
            </TabsTrigger>
            <TabsTrigger
              value="flagged"
              className="data-[state=active]:bg-[#26890d] data-[state=active]:!text-black text-white hover:bg-[#0f3d17] transition-colors duration-200 border border-[#26890d]/60 rounded-md px-2 sm:px-3 md:px-4 py-1 md:py-2 text-xs sm:text-sm md:text-base flex justify-center items-center whitespace-nowrap"
            >
              <span className="sm:hidden data-[state=active]:!text-black">
                Flagged
              </span>
              <span className="hidden sm:block data-[state=active]:!text-black">
                Flagged Reports
              </span>
            </TabsTrigger>
          </TabsList>

          {/* All Reports Tab */}
          <TabsContent value="all" className="mt-0">
            <div className="overflow-x-auto">
              <Table>
                <TableHeader>
                  <TableRow className="border-[#26890d]/30 hover:bg-[#2a2f1e]">
                    <TableHead className="text-[#26890d] font-semibold w-[130px]">
                      Employee ID
                    </TableHead>
                    <TableHead className="text-[#26890d] font-semibold w-[150px] md:w-[170px]">
                      Name
                    </TableHead>
                    <TableHead className="text-[#26890d] font-semibold w-[150px] md:w-[180px]">
                      Position
                    </TableHead>
                    <TableHead className="text-[#26890d] font-semibold w-[100px]">
                      Status
                    </TableHead>
                    {/* <TableHead className="text-center text-[#26890d] font-semibold w-[100px]">
                      Actions
                    </TableHead> */}
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {filteredReports.length > 0 ? (
                    filteredReports.map((report) => (
                      <TableRow
                        key={report.id}
                        className="transition-colors duration-200 border-[#26890d]/30 hover:bg-[#2a2f1e]"
                      >
                        <TableCell className="font-medium text-gray-400">
                          {report.id}
                        </TableCell>
                        <TableCell className="text-white">
                          {report.name}
                        </TableCell>
                        <TableCell className="text-gray-400">
                          {report.position}
                        </TableCell>
                        <TableCell>
                          <Badge
                            variant={report.flagged ? "destructive" : "outline"}
                            className={`transition-all duration-200 pointer-events-none ${
                              report.flagged
                                ? "bg-red-500/20 text-red-500 border-red-500/30"
                                : "bg-[#26890d]/20 text-[#26890d] border-[#26890d]/30"
                            }`}
                          >
                            {report.flagged ? "Flagged" : "Unflagged"}
                          </Badge>
                        </TableCell>
                        {/* <TableCell className="text-right">
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => handleDownload(report.id)}
                            className="px-2 text-[#26890d] border border-[#26890d]/30 hover:bg-[#2a2f1e] hover:text-[#26890d] hover:border-[#26890d]/30 hover:opacity-100"
                          >
                            <View size={14} className="mr-1" /> PDF
                          </Button>
                        </TableCell> */}
                      </TableRow>
                    ))
                  ) : (
                    <TableRow className="border-[#26890d]/30">
                      <TableCell
                        colSpan={5}
                        className="text-center py-6 text-gray-400"
                      >
                        No reports found matching your search
                      </TableCell>
                    </TableRow>
                  )}
                </TableBody>
              </Table>
            </div>
          </TabsContent>

          {/* Unflagged Reports Tab */}
          <TabsContent value="unflagged" className="mt-0">
            <div className="overflow-x-auto">
              <Table>
                <TableHeader>
                  <TableRow className="border-[#26890d]/30 hover:bg-[#2a2f1e]">
                    <TableHead className="text-[#26890d] font-semibold w-[130px]">
                      Employee ID
                    </TableHead>
                    <TableHead className="text-[#26890d] font-semibold w-[150px] md:w-[170px]">
                      Name
                    </TableHead>
                    <TableHead className="text-[#26890d] font-semibold w-[150px] md:w-[180px]">
                      Position
                    </TableHead>
                    <TableHead className="text-[#26890d] font-semibold w-[100px]">
                      Status
                    </TableHead>
                    <TableHead className="text-center text-[#26890d] font-semibold w-[100px]">
                      Actions
                    </TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {filteredReports.length > 0 ? (
                    filteredReports.map((report) => (
                      <TableRow
                        key={report.id}
                        className="transition-colors duration-200 border-[#26890d]/30 hover:bg-[#2a2f1e]"
                      >
                        <TableCell className="font-medium text-gray-400">
                          {report.id}
                        </TableCell>
                        <TableCell className="text-white">
                          {report.name}
                        </TableCell>
                        <TableCell className="text-gray-400">
                          {report.position}
                        </TableCell>
                        <TableCell>
                          <Badge
                            variant="outline"
                            className="transition-all duration-200 pointer-events-none bg-[#26890d]/20 text-[#26890d] border-[#26890d]/30"
                          >
                            Unflagged
                          </Badge>
                        </TableCell>
                        <TableCell className="text-right">
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => handleDownload(report.report_link)}
                            className="px-2 text-[#26890d] border border-[#26890d]/30 hover:bg-[#2a2f1e] hover:text-[#26890d] hover:border-[#26890d]/30 hover:opacity-100"
                          >
                            <View size={14} className="mr-1" /> PDF
                          </Button>
                        </TableCell>
                      </TableRow>
                    ))
                  ) : (
                    <TableRow className="border-[#26890d]/30">
                      <TableCell
                        colSpan={5}
                        className="text-center py-6 text-gray-400"
                      >
                        No unflagged reports found matching your search
                      </TableCell>
                    </TableRow>
                  )}
                </TableBody>
              </Table>
            </div>
          </TabsContent>

          {/* Flagged Reports Tab */}
          <TabsContent value="flagged" className="mt-0">
            <div className="overflow-x-auto">
              <Table>
                <TableHeader>
                  <TableRow className="border-[#26890d]/30 hover:bg-[#2a2f1e]">
                    <TableHead className="text-[#26890d] font-semibold w-[130px]">
                      Employee ID
                    </TableHead>
                    <TableHead className="text-[#26890d] font-semibold w-[150px] md:w-[170px]">
                      Name
                    </TableHead>
                    <TableHead className="text-[#26890d] font-semibold w-[150px] md:w-[180px]">
                      Position
                    </TableHead>
                    <TableHead className="text-[#26890d] font-semibold w-[100px]">
                      Status
                    </TableHead>
                    <TableHead className="text-center text-[#26890d] font-semibold w-[100px]">
                      Actions
                    </TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {filteredReports.length > 0 ? (
                    filteredReports.map((report) => (
                      <TableRow
                        key={report.id}
                        className="transition-colors duration-200 border-[#26890d]/30 hover:bg-[#2a2f1e]"
                      >
                        <TableCell className="font-medium text-gray-400">
                          {report.id}
                        </TableCell>
                        <TableCell className="text-white">
                          {report.name}
                        </TableCell>
                        <TableCell className="text-gray-400">
                          {report.position}
                        </TableCell>
                        <TableCell>
                          <Badge
                            variant="destructive"
                            className="transition-all duration-200 pointer-events-none bg-red-500/20 text-red-500 border-red-500/30"
                          >
                            Flagged
                          </Badge>
                        </TableCell>
                        <TableCell className="text-right">
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => handleDownload(report.report_link)}
                            className="px-2 text-[#26890d] border border-[#26890d]/30 hover:bg-[#2a2f1e] hover:text-[#26890d] hover:border-[#26890d]/30 hover:opacity-100"
                          >
                            <View size={14} className="mr-1" /> PDF
                          </Button>
                        </TableCell>
                      </TableRow>
                    ))
                  ) : (
                    <TableRow className="border-[#26890d]/30">
                      <TableCell
                        colSpan={5}
                        className="text-center py-6 text-gray-400"
                      >
                        No flagged reports found matching your search
                      </TableCell>
                    </TableRow>
                  )}
                </TableBody>
              </Table>
            </div>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
}
