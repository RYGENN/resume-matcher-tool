import React, { useState } from 'react';
import axios from 'axios';

function App() {
  const [jobFile, setJobFile] = useState(null);
  const [resumeFiles, setResumeFiles] = useState([]);
  const [results, setResults] = useState([]);
 
  const uploadJobDescription = async () => {
    const formData = new FormData();
    formData.append('pdf', jobFile);
    await axios.post('http://localhost:5000/upload_job_description', formData);
    alert('Job Description Uploaded!');
  };

  const uploadResumes = async () => {
    const formData = new FormData();
    resumeFiles.forEach(file => formData.append('pdfs', file));
    await axios.post('http://localhost:5000/upload_candidate_resumes', formData);
    alert('Resumes Uploaded!');
  };

  const calculateRanks = async () => {
    const res = await axios.get('http://localhost:5000/calculate_ranks');
    setResults(res.data);
  };

  // sorting function 
  const sortedResults = [...results].sort((a, b) =>{
    return  parseFloat(b.score) - parseFloat(a.score);
  });

  return (
    <div className="root w-full min-h-screen p-6 font-sans flex flex-row">
      <div className="section1 w-1/2  bg-white shadow-md rounded-xl p-6 space-y-8">
        <h1 className="text-2xl font-bold text-center text-blue-700">Resume Matcher Tool</h1>

        {/* Upload JD */}
        <div className="space-y-2">
          <h2 className="text-lg font-semibold">Upload Job Description</h2>
          <input type="file" className="cursor-pointer bg-gray-200 p-2 rounded mr-4" accept="application/pdf" onChange={(e) => setJobFile(e.target.files[0])} />

          <button
            onClick={uploadJobDescription}
            className="cursor-pointer px-4 py-2 bg-blue-600 text-black rounded hover:bg-blue-700"
          >
            Upload JD
          </button>
        </div>

        {/* Upload Resumes */}
        <div className="space-y-2 ">
          <h2 className="text-lg font-semibold">Upload Candidate Resumes</h2>
          <input
            type="file"
            multiple
            accept="application/pdf"
            className='cursor-pointer   bg-gray-200 text-gray-800 rounded p-2 mr-4 shadow-inner hover:bg-gray-300 transition duration-200'
            onChange={(e) => setResumeFiles([...e.target.files])}
          />
          <button
            onClick={uploadResumes}
            className=" cursor-pointer px-4 py-2 bg-green-600 text-black rounded hover:bg-green-700"
          >
            Upload Resumes
          </button>
        </div>

        {/* Calculate */}
        <div className="text-center">
          <button
            onClick={calculateRanks}
            className="cursor-pointer px-6 py-3 bg-green-700 text-black  rounded-lg hover:bg-red-700 mt-4"
          >
            Calculate Rankings
          </button>
        </div>

      
      </div>

      {/* Right Section - Results Display */}
        <div className="w-1/2 p-4">
          <div className="bg-white p-6 rounded-lg shadow-sm h-[calc(100vh-2rem)]">
            <h2 className="text-2xl font-bold mb-6">Resume Rankings</h2>
            
            <div className="overflow-y-auto h-[calc(100%-4rem)] pr-2">
              {sortedResults.length > 0 ? (
                <div className="space-y-4">
                  {sortedResults.map((res, index) => (
                    <div 
                      key={res.filename}
                      className="p-4 border rounded-lg hover:bg-gray-50 transition-colors duration-200"
                    >
                      <div className="flex justify-between items-center">
                        <div className="flex items-center space-x-3">
                          <span className="font-bold text-gray-500">#{index + 1}</span>
                          <span className="font-medium truncate">{res.filename}</span>
                        </div>
                        <span className={`font-bold whitespace-nowrap px-3 py-1 rounded-full ${
                          parseFloat(res.score) >= 70 
                            ? 'bg-green-100 text-green-700' 
                            : parseFloat(res.score) >= 40 
                            ? 'bg-yellow-100 text-yellow-700'
                            : 'bg-red-100 text-red-700'
                        }`}>
                          {Number(res.score).toFixed(2)}%
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center h-full text-gray-500">
                  <svg className="w-12 h-12 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                  </svg>
                  <p className="text-lg font-medium">No results available yet</p>
                  <p className="text-sm">Upload files and calculate rankings to see results</p>
                </div>
              )}
            </div>
          </div>
        </div>
    </div>
  );
}

export default App;
