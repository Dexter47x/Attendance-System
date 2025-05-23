# Face Recognition Attendance System

A modern web-based attendance management system that uses face recognition technology to automate the attendance tracking process.

## Features

- 👤 Face Recognition based attendance marking
- 📝 Easy student registration with photo capture
- 📊 Real-time attendance tracking
- 📱 Responsive web interface
- 📂 CSV-based data storage
- 🎯 Simple and intuitive UI
- 🔄 Automatic face detection and recognition
- 📋 View and manage attendance records

## Technology Stack

- **Backend**: Python Flask
- **Frontend**: HTML5, CSS3, Material Icons
- **Face Recognition**: OpenCV, Haar Cascade Classifier
- **Data Processing**: Pandas, NumPy
- **Storage**: CSV Files
- **Deployment**: Render.com ready

## Prerequisites

- Python 3.8 or higher
- Webcam for face detection
- Modern web browser
- Git

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Dexter47x/Attendance-System.git
   cd Attendance-System
   ```

2. Create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the application:

   ```bash
   python attendance_flask_app/app.py
   ```

5. Open your browser and navigate to:
   ```
   http://localhost:5000
   ```

## Usage

1. **Register Students**:

   - Navigate to the "Register Student" page
   - Enter student details (Enrollment number and Name)
   - Capture student's photo using webcam
   - Click "Register" to save

2. **Train Model**:

   - After registering students, click "Train Model"
   - Wait for the training to complete

3. **Take Attendance**:

   - Go to "Take Attendance" page
   - Enter subject name
   - Start attendance session
   - System will automatically recognize faces and mark attendance

4. **View Records**:
   - Visit "View Attendance" page
   - Check attendance records by date and subject
   - Download or delete attendance records

## Project Structure

```
attendance_flask_app/
├── app.py                 # Main Flask application
├── templates/            # HTML templates
│   ├── index.html
│   ├── register.html
│   ├── attendance.html
│   └── view_attendance.html
├── static/              # Static files (CSS, images)
│   ├── styles.css
│   └── loop_logo.png
├── Attendance/          # Attendance records (CSV)
├── TrainingImage/       # Student photos
└── TrainingImageLabel/  # Trained model data
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenCV for face recognition capabilities
- Flask for the web framework
- Contributors and testers

## Support

For support, please open an issue in the GitHub repository or contact the maintainers.
