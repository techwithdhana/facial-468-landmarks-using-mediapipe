import cv2
import time
import imutils
import mediapipe as mp

cap = cv2.VideoCapture('sources/video2.mp4')
pTime = 0

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)
drawSpec = mpDraw.DrawingSpec(thickness = 1, circle_radius = 1)

while True:
	flag, frame = cap.read()
	frame = imutils.resize(frame, width = 800)

	frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	results = faceMesh.process(frameRGB)
	results = results.multi_face_landmarks

	if results:
		for faceLms in results:
			mpDraw.draw_landmarks(frame, faceLms, mpFaceMesh.FACE_CONNECTIONS, drawSpec, drawSpec)

			for id, lm in enumerate(faceLms.landmark):
                #print(lm)
				ih, iw, ic = frame.shape
				x ,y = int(lm.x*iw), int(lm.y*ih)
				print('id :', id,'co-ordinates x:', x,'and y :', y)

	cTime = time.time()
	fps = 1 / (cTime - pTime)
	pTime = cTime
	cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)

	cv2.imshow('video', frame)
	key = cv2.waitKey(1) & 0xFF
	if key == ord('s'):
		break
		

cap.release()
cv2.destroyAllWindows()
