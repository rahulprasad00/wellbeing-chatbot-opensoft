"use client";

import React, { useState, useRef } from "react";
import {
  Upload,
  X,
  AlertCircle,
  CheckCircle,
  FileText,
  FilePlus,
  Loader2,
  UploadCloud,
  Database,
  Edit,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectTrigger,
  SelectValue,
  SelectContent,
  SelectItem,
} from "@/components/ui/select";
import { Label } from "@/components/ui/label";
import { cn } from "@/lib/utils";
import axios from "axios";
import { toast } from "sonner";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { server } from "@/utils";

interface FileUploadStatus {
  id: string;
  name: string;
  size: string;
  status: "queued" | "success" | "error" | "uploading";
  message?: string;
  timestamp: Date;
  file: File; // Store the actual file reference
  targetTable: string; // Store the target table for each file
}

interface EditFileDialogProps {
  file: FileUploadStatus | null;
  tables: { id: string; label: string }[];
  isOpen: boolean;
  onClose: () => void;
  onSave: (fileId: string, newTable: string) => void;
}

function EditFileDialog({
  file,
  tables,
  isOpen,
  onClose,
  onSave,
}: EditFileDialogProps) {
  const [selectedTable, setSelectedTable] = useState<string>(
    file?.targetTable || ""
  );

  React.useEffect(() => {
    if (file) {
      setSelectedTable(file.targetTable);
    }
  }, [file]);

  const handleSave = () => {
    if (file) {
      onSave(file.id, selectedTable);
      onClose();
    }
  };

  return (
    <Dialog open={isOpen} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="bg-hr-black border-[#26890d]/30 text-white">
        <DialogHeader>
          <DialogTitle className="text-[#26890d]">Edit File Target</DialogTitle>
          <DialogDescription className="text-gray-400">
            Change the destination table for this file
          </DialogDescription>
        </DialogHeader>

        {file && (
          <div className="py-4">
            <p className="text-sm mb-4">
              <span className="font-medium">File:</span> {file.name}
            </p>
            <Label
              htmlFor="edit-table-select"
              className="text-[#26890d] mb-2 block"
            >
              Select Target Table
            </Label>
            <Select value={selectedTable} onValueChange={setSelectedTable}>
              <SelectTrigger
                id="edit-table-select"
                className="w-full bg-hr-black border-[#26890d]/30 focus:border-[#26890d] focus:ring-[#26890d]/20"
              >
                <SelectValue placeholder="Select a table to import data" />
              </SelectTrigger>
              <SelectContent className="bg-hr-black border-[#26890d]/30">
                {tables.map((table) => (
                  <SelectItem
                    key={table.id}
                    value={table.id}
                    className="focus:bg-[#26890d]/20 focus:text-white"
                  >
                    {table.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        )}

        <DialogFooter>
          <Button
            variant="outline"
            onClick={onClose}
            className="border-[#26890d]/30 text-white"
          >
            Cancel
          </Button>
          <Button
            onClick={handleSave}
            className="bg-[#26890d]/80 hover:bg-[#26890d] text-white"
            disabled={!selectedTable}
          >
            Save Changes
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

export function UploadData() {
  const [selectedTable, setSelectedTable] = useState<string>("");
  const [dragActive, setDragActive] = useState<boolean>(false);
  const [uploadedFiles, setUploadedFiles] = useState<FileUploadStatus[]>([]);
  const [uploading, setUploading] = useState<boolean>(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [editingFile, setEditingFile] = useState<FileUploadStatus | null>(null);
  const [isEditDialogOpen, setIsEditDialogOpen] = useState(false);

  // Available tables from backend models
  const tables = [
    { id: "master", label: "Master (Employees)" },
    { id: "vibemeter", label: "Vibe Meter" },
    { id: "activity_tracker", label: "Activity Tracker" },
    { id: "leave", label: "Leave Records" },
    { id: "onboarding", label: "Onboarding" },
    { id: "performance", label: "Performance" },
    { id: "rewards", label: "Rewards" },
    { id: "conversation", label: "Conversations" },
    { id: "message", label: "Messages" },
  ];

  // Get table label by id
  const getTableLabel = (tableId: string): string => {
    const table = tables.find((t) => t.id === tableId);
    return table ? table.label : tableId;
  };

  // Handle drag events
  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();

    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  // Triggers when file is dropped
  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0] && selectedTable) {
      handleFiles(e.dataTransfer.files);
    }
  };

  // Triggers when file is selected with click
  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0] && selectedTable) {
      handleFiles(e.target.files);
    }
  };

  // Add files to queue with the currently selected table
  const handleFiles = (files: FileList) => {
    if (!selectedTable) {
      toast.error("Table selection required", {
        description: "Please select a table before uploading files",
      });
      return;
    }

    // Add files to queue with the current target table
    const newFiles: FileUploadStatus[] = Array.from(files).map((file) => ({
      id: Math.random().toString(36).substring(2, 9),
      name: file.name,
      size: formatFileSize(file.size),
      status: "queued",
      timestamp: new Date(),
      file: file,
      targetTable: selectedTable, // Store the currently selected table with the file
    }));

    setUploadedFiles((prev) => [...newFiles, ...prev]);

    // Show notification
    toast.success("Files added to queue", {
      description: `${newFiles.length} files added to queue for ${getTableLabel(
        selectedTable
      )}. Click "Upload Files" to start uploading.`,
    });
  };

  // Open edit dialog for a file
  const handleEditFile = (file: FileUploadStatus) => {
    setEditingFile(file);
    setIsEditDialogOpen(true);
  };

  // Save table change for a file
  const handleSaveTableChange = (fileId: string, newTable: string) => {
    setUploadedFiles((prev) =>
      prev.map((f) =>
        f.id === fileId
          ? {
              ...f,
              targetTable: newTable,
            }
          : f
      )
    );

    toast.success("Target table updated", {
      description: `File will now be uploaded to ${getTableLabel(newTable)}`,
    });
  };

  // Upload all queued files to their respective tables
  const uploadQueuedFiles = async () => {
    const queuedFiles = uploadedFiles.filter((f) => f.status === "queued");

    if (queuedFiles.length === 0) {
      toast.info("No files to upload", {
        description: "Add files to the queue first",
      });
      return;
    }

    setUploading(true);

    // Track uploads per table for summary
    const tableUploadResults: Record<
      string,
      { success: number; error: number }
    > = {};

    // Process each queued file with its own target table
    for (const fileItem of queuedFiles) {
      // Initialize tableUploadResults for this table if needed
      if (!tableUploadResults[fileItem.targetTable]) {
        tableUploadResults[fileItem.targetTable] = { success: 0, error: 0 };
      }

      // Update status to uploading
      setUploadedFiles((prev) =>
        prev.map((f) =>
          f.id === fileItem.id
            ? {
                ...f,
                status: "uploading",
              }
            : f
        )
      );

      const formData = new FormData();
      formData.append("file", fileItem.file);
      formData.append("table", fileItem.targetTable); // Use the file's target table

      try {
        console.log("Sending file to table:", fileItem.targetTable); // Debug log

        const response = await axios.post(
          `${server}/api/data/ingest?table=${fileItem.targetTable}`, // Add as URL parameter too
          formData,
          {
            headers: {
              "Content-Type": "multipart/form-data",
              Authorization: `Bearer ${localStorage.getItem("access_token")}`,
            },
          }
        );

        // Update file status to success
        setUploadedFiles((prev) =>
          prev.map((f) =>
            f.id === fileItem.id
              ? {
                  ...f,
                  status: "success",
                  message: response.data.message || "Upload successful",
                }
              : f
          )
        );

        // Count successful uploads for this table
        tableUploadResults[fileItem.targetTable].success++;
      } catch (error) {
        console.error("Upload error:", error);
        let errorMessage = "Unknown error occurred";

        if (axios.isAxiosError(error) && error.response) {
          errorMessage = error.response.data.detail || "Upload failed";
        }

        // Update file status to error
        setUploadedFiles((prev) =>
          prev.map((f) =>
            f.id === fileItem.id
              ? {
                  ...f,
                  status: "error",
                  message: errorMessage,
                }
              : f
          )
        );

        // Count failed uploads for this table
        tableUploadResults[fileItem.targetTable].error++;
      }
    }

    setUploading(false);

    // Show summary toast for each table
    const tableEntries = Object.entries(tableUploadResults);
    if (tableEntries.length === 1) {
      // Single table summary
      const [tableId, results] = tableEntries[0];
      if (results.error === 0) {
        toast.success(`Upload to ${getTableLabel(tableId)} complete`, {
          description: `Successfully uploaded ${results.success} files`,
        });
      } else {
        toast.error(
          `Upload to ${getTableLabel(tableId)} completed with errors`,
          {
            description: `${results.success} successful, ${results.error} failed. Check the list for details.`,
          }
        );
      }
    } else {
      // Multiple tables summary
      const totalSuccess = Object.values(tableUploadResults).reduce(
        (sum, result) => sum + result.success,
        0
      );
      const totalError = Object.values(tableUploadResults).reduce(
        (sum, result) => sum + result.error,
        0
      );

      if (totalError === 0) {
        toast.success("All uploads completed successfully", {
          description: `Uploaded ${totalSuccess} files to ${tableEntries.length} different tables`,
        });
      } else {
        toast.error("Uploads completed with errors", {
          description: `${totalSuccess} successful, ${totalError} failed across ${tableEntries.length} tables. Check the list for details.`,
        });
      }
    }
  };

  // Format file size to human-readable format
  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
  };

  // Open file browser
  const onButtonClick = () => {
    if (!selectedTable) {
      toast.error("Table selection required", {
        description: "Please select a table before uploading files",
      });
      return;
    }
    fileInputRef.current?.click();
  };

  return (
    <div id="upload-data-section" className="hr-panel p-6 rounded-xl mt-8">
      <div className="mb-6">
        <h2 className="hr-report-title text-2xl flex items-center mb-2">
          <Upload className="h-6 w-6 mr-3 text-[#26890d]" />
          Upload Data
        </h2>
        <p className="text-gray-400">
          Import data files for HR analytics and reporting
        </p>
      </div>

      <div className="mt-6">
        {/* Table Selection */}
        <div className="mb-6">
          <Label htmlFor="table-select" className="text-[#26890d] mb-2 block">
            Select Target Table
          </Label>
          <Select value={selectedTable} onValueChange={setSelectedTable}>
            <SelectTrigger
              id="table-select"
              className="w-full bg-hr-black border-[#26890d]/30 focus:border-[#26890d] focus:ring-[#26890d]/20"
            >
              <SelectValue placeholder="Select a table to import data" />
            </SelectTrigger>
            <SelectContent className="bg-hr-black border-[#26890d]/30">
              {tables.map((table) => (
                <SelectItem
                  key={table.id}
                  value={table.id}
                  className="focus:bg-[#26890d]/20 focus:text-white"
                >
                  {table.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {/* Upload Area */}
        <div
          className={cn(
            "border-2 border-dashed rounded-lg p-8 text-center transition-all duration-200",
            dragActive
              ? "border-[#26890d] bg-[#26890d]/10"
              : "border-[#26890d]/30 hover:border-[#26890d]/60",
            !selectedTable && "opacity-70 pointer-events-none"
          )}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
        >
          <input
            ref={fileInputRef}
            className="hidden"
            type="file"
            accept=".csv,.xlsx,.json"
            onChange={handleChange}
            disabled={!selectedTable || uploading}
            multiple
          />

          <div className="flex flex-col items-center justify-center">
            <FilePlus className="h-16 w-16 text-[#26890d]/70 mb-4" />
            <p className="text-lg font-medium text-white mb-2">
              {uploading ? "Processing files..." : "Drag & Drop Files Here"}
            </p>
            <p className="text-sm text-gray-400 mb-4">
              Files will be queued for upload to{" "}
              <span className="font-medium text-[#26890d]">
                {selectedTable
                  ? getTableLabel(selectedTable)
                  : "selected table"}
              </span>
            </p>
            <Button
              onClick={onButtonClick}
              className="bg-[#26890d]/80 hover:bg-[#26890d] text-white"
              disabled={!selectedTable || uploading}
            >
              {uploading ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Processing...
                </>
              ) : (
                "Select Files"
              )}
            </Button>
          </div>
        </div>

        {/* Upload Queue Section */}
        {uploadedFiles.filter((f) => f.status === "queued").length > 0 && (
          <div className="mt-6 p-4 border border-[#26890d]/30 bg-[#26890d]/5 rounded-lg">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center">
                <UploadCloud className="h-5 w-5 text-[#26890d] mr-2" />
                <span className="text-white font-medium">
                  {uploadedFiles.filter((f) => f.status === "queued").length}{" "}
                  files ready to upload
                </span>
              </div>
              <Button
                onClick={uploadQueuedFiles}
                className="bg-[#26890d] hover:bg-[#26890d]/90 text-white"
                disabled={
                  uploading ||
                  uploadedFiles.filter((f) => f.status === "queued").length ===
                    0
                }
              >
                {uploading ? (
                  <>
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    Uploading...
                  </>
                ) : (
                  "Upload Files"
                )}
              </Button>
            </div>
            <p className="text-xs text-gray-400">
              You can change the target table for each file before uploading
            </p>
          </div>
        )}

        {/* Uploaded Files List */}
        {uploadedFiles.length > 0 && (
          <div className="mt-8">
            <h3 className="text-[#26890d] text-lg font-medium mb-4">Files</h3>
            <div className="space-y-3 max-h-[300px] overflow-y-auto hr-scrollbar pr-2">
              {uploadedFiles.map((file) => (
                <div
                  key={file.id}
                  className={cn(
                    "flex items-center justify-between p-3 rounded-lg border",
                    file.status === "queued"
                      ? "border-yellow-500/30 bg-yellow-500/5"
                      : file.status === "uploading"
                      ? "border-[#26890d]/30 bg-hr-black/60"
                      : file.status === "success"
                      ? "border-[#26890d]/30 bg-[#26890d]/10"
                      : "border-red-500/30 bg-red-500/10"
                  )}
                >
                  <div className="flex items-center flex-1">
                    <div className="mr-3">
                      {file.status === "queued" ? (
                        <FileText className="h-5 w-5 text-yellow-500" />
                      ) : file.status === "uploading" ? (
                        <Loader2 className="h-5 w-5 animate-spin text-[#26890d]" />
                      ) : file.status === "success" ? (
                        <CheckCircle className="h-5 w-5 text-[#26890d]" />
                      ) : (
                        <AlertCircle className="h-5 w-5 text-red-500" />
                      )}
                    </div>
                    <div className="flex-1">
                      <p className="text-sm font-medium text-white">
                        {file.name}
                      </p>
                      <div className="flex items-center flex-wrap text-xs text-gray-400 mt-1">
                        <span>{file.size}</span>
                        <span className="mx-2">•</span>
                        <span>
                          <Database className="inline h-3 w-3 mr-1 opacity-70" />
                          {getTableLabel(file.targetTable)}
                        </span>
                        <span className="mx-2">•</span>
                        <span>
                          {file.timestamp.toLocaleTimeString([], {
                            hour: "2-digit",
                            minute: "2-digit",
                          })}
                        </span>
                        {file.status === "queued" && (
                          <>
                            <span className="mx-2">•</span>
                            <span className="text-yellow-500">
                              Queued for upload
                            </span>
                          </>
                        )}
                        {file.status === "uploading" && (
                          <>
                            <span className="mx-2">•</span>
                            <span className="text-[#26890d]">Uploading...</span>
                          </>
                        )}
                        {file.status === "error" && file.message && (
                          <>
                            <span className="mx-2">•</span>
                            <span className="text-red-400">{file.message}</span>
                          </>
                        )}
                        {file.status === "success" && file.message && (
                          <>
                            <span className="mx-2">•</span>
                            <span className="text-[#26890d]/90">
                              {file.message}
                            </span>
                          </>
                        )}
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center">
                    {file.status === "queued" && (
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-8 w-8 rounded-full hover:bg-[#26890d]/10 mr-1"
                        onClick={() => handleEditFile(file)}
                        disabled={uploading}
                        title="Edit target table"
                      >
                        <Edit className="h-4 w-4" />
                      </Button>
                    )}
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-8 w-8 rounded-full hover:bg-[#26890d]/10"
                      onClick={() =>
                        setUploadedFiles((prev) =>
                          prev.filter((f) => f.id !== file.id)
                        )
                      }
                      disabled={file.status === "uploading"}
                      title="Remove file"
                    >
                      <X className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      <div className="mt-6 flex justify-between">
        <div className="text-sm text-gray-400">
          {uploadedFiles.filter((f) => f.status === "success").length} files
          successfully imported
        </div>
        <Button
          variant="outline"
          className="border-[#26890d]/30 text-[#26890d] hover:bg-[#26890d]/10"
          onClick={() => setUploadedFiles([])}
          disabled={uploadedFiles.length === 0 || uploading}
        >
          Clear History
        </Button>
      </div>

      {/* Edit File Dialog */}
      <EditFileDialog
        file={editingFile}
        tables={tables}
        isOpen={isEditDialogOpen}
        onClose={() => {
          setIsEditDialogOpen(false);
          setEditingFile(null);
        }}
        onSave={handleSaveTableChange}
      />
    </div>
  );
}
