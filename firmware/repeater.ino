#include <Servo.h>
#include <NeoSWSerial.h>

// Pins
static const uint8_t SERVO1_PIN = 11;
static const uint8_t SERVO2_PIN = 12;
static const uint8_t GRBL_RX_PIN = 8;  // from GRBL TX0
static const uint8_t GRBL_TX_PIN = 9;  // to   GRBL RX0

// Serial
NeoSWSerial grbl(GRBL_RX_PIN, GRBL_TX_PIN);
static const unsigned long USB_BAUD  = 115200;
static const unsigned long GRBL_BAUD = 9600;

// Servos
Servo s1, s2;

// Line buffer
char lineBuf[96];          // big enough for typical GRBL lines
uint8_t lineLen = 0;
bool sawCR = false;        // track CRLF to avoid double-send

// Helpers
static inline char leadNonSpace(const char* s, uint8_t n){
  for (uint8_t i=0;i<n;++i){ char c=s[i]; if (c!=' ' && c!='\t' && c!='\r') return c; }
  return 0;
}
static inline void doServo(const char* s, uint8_t n){
  // parse: "<servo> <angle>"
  int servo=-1, angle=-1;
  // simple parse without String:
  int i=0; while (i<n && s[i]==' ') i++;
  int start=i; while (i<n && s[i]>='0' && s[i]<='9') i++;
  if (i>start) servo = atoi(&s[start]);
  while (i<n && s[i]==' ') i++;
  start=i; while (i<n && s[i]>='0' && s[i]<='9') i++;
  if (i>start) angle = atoi(&s[start]);
  if (servo>=1 && servo<=2 && angle>=0){
    if (angle>180) angle=180;
    if (servo==1) s1.write(angle); else s2.write(angle);
  }
}
static inline void forwardToGrblOnce(){
  if (lineLen==0) return;
  char L = leadNonSpace(lineBuf, lineLen);
  if (L=='G' || L=='g' || L=='$'){
    // send exactly one newline
    grbl.write(lineBuf, lineLen);
    grbl.write('\n');
  } else {
    doServo(lineBuf, lineLen);   // act silently
  }
  lineLen = 0;
}

void setup(){
  Serial.begin(USB_BAUD);
  grbl.begin(GRBL_BAUD);

  s1.attach(SERVO1_PIN);
  s2.attach(SERVO2_PIN);
  s1.write(90); s2.write(90);

  // swallow GRBL boot junk briefly
  unsigned long t0 = millis();
  while (millis() - t0 < 200) { while (grbl.available()) (void)grbl.read(); }
}

void loop(){
  // USB -> (GRBL | Servos)
  while (Serial.available()){
    char c = (char)Serial.read();

    if (c == '\r'){
      // remember CR; don't send yet (wait to see if LF follows)
      sawCR = true;
      continue;
    }
    if (c == '\n'){
      // If CRLF, we arrive here once per line; if bare LF, also here.
      sawCR = false;
      forwardToGrblOnce();
      continue;
    }
    // Any other char clears CR flag
    sawCR = false;

    if (lineLen < sizeof(lineBuf)-1){
      lineBuf[lineLen++] = c;
    } else {
      // overflow: just drop extras to avoid split lines
    }
  }

  // GRBL -> USB (raw passthrough)
  while (grbl.available()){
    Serial.write(grbl.read());
  }
}