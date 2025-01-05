import React, { useState } from 'react';
import { Box, Typography, Button, TextField, CircularProgress, Snackbar } from '@mui/material';
import MuiAlert from '@mui/material/Alert';

const App = () => {
  const [image, setImage] = useState(null);
  const [breed, setBreed] = useState(null);
  const [userFeedback, setUserFeedBack] = useState(null);
  const [correctBreed, setCorrectBreed] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [openSnackbar, setOpenSnackbar] = useState(false);

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.readAsDataURL(file);
      reader.onloadend = () => {
        setImage(reader?.result);
      };
    }
  };

  const handleSubmit = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await fetch('/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ image, lang: 'eng' }),
      });

      if (!response.ok) {
        throw new Error('Failed to get prediction. Please try again later.');
      }

      const data = await response.json();
      setBreed(data?.predicted_breed);
      setUserFeedBack(null);
    } catch (error) {
      setError(error.message);
      setOpenSnackbar(true);
    } finally {
      setIsLoading(false);
    }
  };

  const handleFeedback = async (isCorrect) => {
    if (isCorrect) {
      setUserFeedBack('Thank you for confirming!');
    } else {
      setUserFeedBack('We are sorry...Please provide correct breed name:');
    }
  };

  const handleCorrectBreedSubmit = async () => {
    setError(null);
    setUserFeedBack("Thank you for improving our model!");
  };

  const handleFileButtonClick = () => {
    document.getElementById('file-input').click();
  };

  const handleCloseSnackbar = () => {
    setOpenSnackbar(false);
  };

  return (
    <Box
      display="flex"
      flexDirection="column"
      alignItems="center"
      justifyContent="center"
      minHeight="100vh"
      textAlign="center"
      p={2}
    >
      {/* Title */}
      <Typography variant="h3" gutterBottom sx={{ paddingBottom: '20px' }}>
        Dog Breed Classifier
      </Typography>

      {/* Select Image Button */}
      <Button variant="contained" color="primary" onClick={handleFileButtonClick} sx={{ marginBottom: '20px' }}>
        Select Image
      </Button>

      {/* Hidden File Input */}
      <input
        id="file-input"
        type="file"
        accept="image/*"
        onChange={handleImageChange}
        style={{ display: 'none' }}
      />

      {/* Display Image */}
      {image && (
        <img
          src={image}
          alt="Selected"
          style={{
            height: '40vh',
            objectFit: 'contain',
            marginBottom: '20px',
          }}
        />
      )}

      {/* Submit Button */}
      {image && !isLoading && (
        <Button variant="contained" color="primary" onClick={handleSubmit} sx={{ marginBottom: '20px' }}>
          Submit Image
        </Button>
      )}

      {isLoading && <CircularProgress />}

      {/* Predicted Breed */}
      {breed && !isLoading && <Typography variant="h6" sx={{ marginBottom: '20px' }}>Predicted breed: {breed}</Typography>}

      {/* Feedback Buttons */}
      {breed && !userFeedback && !isLoading && (
        <Box>
          <Button
            variant="outlined"
            color="success"
            onClick={() => handleFeedback(true)}
            sx={{ marginRight: '10px' }}
          >
            You are right
          </Button>
          <Button variant="outlined" color="error" onClick={() => handleFeedback(false)}>
            You are wrong
          </Button>
        </Box>
      )}

      {/* Correct Breed Input */}
      {userFeedback === 'We are sorry...Please provide correct breed name:' && (
        <>
          <TextField
            label="Enter correct breed name"
            variant="outlined"
            value={correctBreed}
            onChange={(e) => setCorrectBreed(e.target.value)}
            sx={{ marginBottom: '20px' }}
          />
          <Button
            variant="contained"
            color="primary"
            onClick={handleCorrectBreedSubmit}
            disabled={!correctBreed}
          >
            Submit Correct Breed
          </Button>
        </>
      )}

      {/* Error Snackbar */}
      <Snackbar
        open={openSnackbar}
        autoHideDuration={6000}
        onClose={handleCloseSnackbar}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
      >
        <MuiAlert onClose={handleCloseSnackbar} severity="error" sx={{ width: '100%' }}>
          {error}
        </MuiAlert>
      </Snackbar>
    </Box>
  );
};

export default App;