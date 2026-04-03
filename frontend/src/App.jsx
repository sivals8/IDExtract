import { useCallback, useEffect, useState } from "react";
import { useDropzone } from "react-dropzone";

export default function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const onDrop = useCallback((acceptedFiles) => {
    const selectedFile = acceptedFiles?.[0];
    if (!selectedFile) return;

    setFile(selectedFile);
    setPreview(URL.createObjectURL(selectedFile));
    setResult(null);
    setError("");
  }, []);

  //allow the drag and drop feature using react-dropzone package
  const { getRootProps, getInputProps, isDragActive, open } = useDropzone({
    onDrop,
    accept: {
      "image/*": [],
    },
    multiple: false,
    noClick: true,
  });

  useEffect(() => {
    return () => {
      if (preview) {
        URL.revokeObjectURL(preview);
      }
    };
  }, [preview]);

  const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";
  const handleUpload = async () => {
    if (!file) {
      setError("Please choose an image first.");
      return;
    }

    setLoading(true);
    setError("");

    const formData = new FormData();
    formData.append("file", file);
    try {
      const response = await fetch(`${API_URL}/extract`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError("Upload failed. Make sure the backend is running.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#15141f] p-8 text-white">
      <div className="mx-auto max-w-3xl">
        <h1 className="mb-4 text-3xl font-bold">IDExtract</h1>
        <p className="mb-6 text-gray-300">
          Upload an image and extract text from it.
        </p>

        <div
          {...getRootProps()}
          className={`mb-4 cursor-pointer rounded-xl border-2 border-dashed p-8 text-center transition ${
            isDragActive
              ? "border-teal-400 bg-teal-900/20"
              : "border-gray-600 bg-[#1c1b29] hover:border-teal-600"
          }`}
        >
          <input {...getInputProps()} />

          <p className="mb-3 text-lg font-medium">
            {isDragActive
              ? "Drop the image here..."
              : "Drag and drop an image here"}
          </p>

          <p className="mb-4 text-sm text-gray-400">
            or click the button below to browse
          </p>

          <button
            type="button"
            onClick={open}
            className="rounded-xl bg-teal-800 px-5 py-2 text-white transition hover:bg-teal-900"
          >
            Choose Image
          </button>

          {file && (
            <p className="mt-4 text-sm text-gray-300">
              Selected file: {file.name}
            </p>
          )}
        </div>

        <button
          onClick={handleUpload}
          className="rounded-xl bg-teal-800 px-5 py-2 text-white transition hover:bg-teal-900 disabled:opacity-50"
          disabled={loading}
        >
          {loading ? "Processing..." : "Extract Info"}
        </button>

        {error && <p className="mt-4 text-red-500">{error}</p>}

        {preview && (
          <div className="mt-6">
            <h2 className="mb-2 text-xl font-semibold">Preview</h2>
            <img
              src={preview}
              alt="Preview"
              className="max-h-80 rounded-lg border border-gray-600"
            />
          </div>
        )}

        {result && (
          <div className="mt-6 rounded-xl border border-gray-600 p-4">
            <h2 className="mb-2 text-xl font-semibold">Results</h2>
            <p><strong>Name:</strong> {result.name || "Not found"}</p>
            <p><strong>Designation:</strong> {result.designation || "Not found"}</p>
            <p><strong>ID Number:</strong> {result.id_number || "Not found"}</p>
            <p><strong>Issued Date:</strong> {result.issued_date || "Not found"}</p>
            <p><strong>Expiry Date:</strong> {result.expiry_date || "Not found"}</p>

            <div className="mt-4">
              <h3 className="font-semibold">Raw OCR Text</h3>
              <pre className="mt-2 whitespace-pre-wrap rounded-lg bg-[#1c1b29] p-3 text-sm">
                {result.raw_text || "No text returned."}
              </pre>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}