
import React, { useState, useMemo } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';

const employees = [
  { id: 1, name: 'Alex Johnson', position: 'Senior Developer', flagged: false },
  { id: 2, name: 'Sarah Williams', position: 'Product Manager', flagged: true },
  { id: 3, name: 'Michael Chen', position: 'UI/UX Designer', flagged: false },
  { id: 4, name: 'Emily Davis', position: 'Marketing Specialist', flagged: false },
  { id: 5, name: 'Robert Taylor', position: 'System Analyst', flagged: true },
];

interface EmployeeTableProps {
  searchQuery?: string;
}

export function EmployeeTable({ searchQuery = '' }: EmployeeTableProps) {
  const [hoveredRow, setHoveredRow] = useState<number | null>(null);

  const filteredEmployees = useMemo(() => {
    if (!searchQuery) return employees;
    
    const lowerCaseQuery = searchQuery.toLowerCase();
    return employees.filter(
      employee => 
        employee.id.toString().includes(lowerCaseQuery) || 
        employee.name.toLowerCase().includes(lowerCaseQuery)
    );
  }, [searchQuery]);

  const filteredFlaggedEmployees = useMemo(() => {
    return filteredEmployees.filter(e => e.flagged);
  }, [filteredEmployees]);

  const filteredUnflaggedEmployees = useMemo(() => {
    return filteredEmployees.filter(e => !e.flagged);
  }, [filteredEmployees]);

  return (
    <Card className="shadow-card bg-card border border-hr-green/20">
      <CardHeader className="pb-2">
        <CardTitle className="text-xl font-medium text-hr-green">Employee Status</CardTitle>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="all" className="w-full">
          <TabsList className="w-full mb-4 grid grid-cols-2">
            <TabsTrigger 
              value="all" 
              className="data-[state=active]:bg-hr-green data-[state=active]:text-black"
            >
              All Employees
            </TabsTrigger>
            <TabsTrigger 
              value="flagged" 
              className="data-[state=active]:bg-hr-green data-[state=active]:text-black"
            >
              Flagged Employees
            </TabsTrigger>
          </TabsList>
          
          <TabsContent value="all" className="mt-0">
            <Table>
              <TableHeader>
                <TableRow className="border-hr-green/20">
                  <TableHead className="w-[50px] text-hr-green">ID</TableHead>
                  <TableHead className="text-hr-green">Name</TableHead>
                  <TableHead className="text-hr-green">Position</TableHead>
                  <TableHead className="text-right text-hr-green">Status</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {filteredEmployees.length > 0 ? (
                  filteredEmployees.map((employee) => (
                    <TableRow 
                      key={employee.id}
                      className="transition-colors duration-200 border-hr-green/10"
                      onMouseEnter={() => setHoveredRow(employee.id)}
                      onMouseLeave={() => setHoveredRow(null)}
                      style={{
                        backgroundColor: hoveredRow === employee.id ? 'rgba(134, 188, 37, 0.1)' : 'transparent'
                      }}
                    >
                      <TableCell className="font-medium text-muted-foreground">{employee.id}</TableCell>
                      <TableCell className="text-white">{employee.name}</TableCell>
                      <TableCell className="text-muted-foreground">{employee.position}</TableCell>
                      <TableCell className="text-right">
                        <Badge 
                          variant={employee.flagged ? "destructive" : "outline"}
                          className={`
                            transition-all duration-200 
                            ${employee.flagged ? 'bg-destructive/20 text-destructive border-destructive/30' : 'bg-hr-green/20 text-hr-green border-hr-green/30'}
                          `}
                        >
                          {employee.flagged ? 'Flagged' : 'Unflagged'}
                        </Badge>
                      </TableCell>
                    </TableRow>
                  ))
                ) : (
                  <TableRow>
                    <TableCell colSpan={4} className="text-center py-6 text-muted-foreground">
                      No employees found matching your search
                    </TableCell>
                  </TableRow>
                )}
              </TableBody>
            </Table>
          </TabsContent>
          
          <TabsContent value="flagged" className="mt-0">
            <Table>
              <TableHeader>
                <TableRow className="border-hr-green/20">
                  <TableHead className="w-[50px] text-hr-green">ID</TableHead>
                  <TableHead className="text-hr-green">Name</TableHead>
                  <TableHead className="text-hr-green">Position</TableHead>
                  <TableHead className="text-right text-hr-green">Status</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {filteredFlaggedEmployees.length > 0 ? (
                  filteredFlaggedEmployees.map((employee) => (
                    <TableRow 
                      key={employee.id}
                      className="transition-colors duration-200 border-hr-green/10"
                      onMouseEnter={() => setHoveredRow(employee.id)}
                      onMouseLeave={() => setHoveredRow(null)}
                      style={{
                        backgroundColor: hoveredRow === employee.id ? 'rgba(134, 188, 37, 0.1)' : 'transparent'
                      }}
                    >
                      <TableCell className="font-medium text-muted-foreground">{employee.id}</TableCell>
                      <TableCell className="text-white">{employee.name}</TableCell>
                      <TableCell className="text-muted-foreground">{employee.position}</TableCell>
                      <TableCell className="text-right">
                        <Badge 
                          variant="destructive"
                          className="bg-destructive/20 text-destructive border-destructive/30"
                        >
                          Flagged
                        </Badge>
                      </TableCell>
                    </TableRow>
                  ))
                ) : (
                  <TableRow>
                    <TableCell colSpan={4} className="text-center py-6 text-muted-foreground">
                      No flagged employees found matching your search
                    </TableCell>
                  </TableRow>
                )}
              </TableBody>
            </Table>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
}
