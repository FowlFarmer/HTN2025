; Robotic Painting G-code
; Generated: 2025-09-13T19:11:38.660032
; Total strokes: 4
; Estimated time: 6.2s

G21 ; Set units to millimeters
G90 ; Absolute positioning
M3 ; Prepare pen/brush system
G0 Z15 ; Lift pen to safe height
G0 X80 Y80 ; Move to canvas center

; Stroke 0: boundary (warm)
; Length: 10.0mm, Speed: 10.0mm/s
G0 X10.00 Y10.00 ; Move to start
G0 Z0 ; Lower pen
G1 X10.00 Y10.00 F600
G1 X15.00 Y10.00 F600
G1 X15.00 Y15.00 F600
G0 Z15 ; Lift pen

; Stroke 2: boundary (cool)
; Length: 20.0mm, Speed: 10.0mm/s
G0 X50.00 Y50.00 ; Move to start
G0 Z0 ; Lower pen
G1 X50.00 Y50.00 F600
G1 X55.00 Y50.00 F600
G1 X55.00 Y55.00 F600
G1 X50.00 Y55.00 F600
G1 X50.00 Y50.00 F600
G0 Z15 ; Lift pen

; Stroke 1: internal (warm)
; Length: 7.1mm, Speed: 20.0mm/s
G0 X20.00 Y20.00 ; Move to start
G0 Z0 ; Lower pen
G1 X20.00 Y20.00 F1200
G1 X25.00 Y25.00 F1200
G0 Z15 ; Lift pen

; Stroke 3: detail (cool)
; Length: 1.0mm, Speed: 20.0mm/s
G0 X52.50 Y52.50 ; Move to start
G0 Z0 ; Lower pen
G1 X52.50 Y52.50 F1200
G1 X52.50 Y53.50 F1200
G0 Z15 ; Lift pen

G0 X0 Y0 ; Return to origin
M5 ; Turn off pen/brush system
; End of program