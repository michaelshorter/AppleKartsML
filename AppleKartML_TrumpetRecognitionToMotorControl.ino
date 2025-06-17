/*
17th June 2025 - Mike Shorter

This code is designed for an Arduino Nano BLE Sense and DollaTek Dual Channel L298N PWM motor controller. 
One note turns both motors clockwise, another both motors counterclockwise, 
another just one motor and the final not the other motor.

The threshholds can all be fine tuned blow

*/

#define EIDSP_QUANTIZE_FILTERBANK   0
#define EI_CLASSIFIER_SLICES_PER_MODEL_WINDOW 4

#include <PDM.h>
#include <AppleKarts_ML_inferencing.h>

// === Motor Pins ===
const int motorA_IN1 = 2;
const int motorA_IN2 = 3;
const int motorB_IN3 = 4;
const int motorB_IN4 = 5;

// === Timed Motor Settings (blue & high red only) ===
unsigned long motorRunTime = 300;    // Duration motor runs (ms)
unsigned long motorWaitTime = 1000;  // Cooldown before next run (ms)

// === Per-Label Confidence Thresholds ===
const float THRESH_BLUE     = 0.9;
const float THRESH_HIGHRED  = 0.99;
const float THRESH_LOWRED   = 0.7;
const float THRESH_PINK     = 0.7;

/** Audio inference buffers */
typedef struct {
    signed short *buffers[2];
    unsigned char buf_select;
    unsigned char buf_ready;
    unsigned int buf_count;
    unsigned int n_samples;
} inference_t;

static inference_t inference;
static bool record_ready = false;
static signed short *sampleBuffer;
static bool debug_nn = false;

// === Motor FSM (blue/high red only) ===
enum MotorState { IDLE, RUNNING, WAITING };
MotorState motorState = IDLE;
unsigned long motorStartTime = 0;
String activeTimedLabel = "";

// === Track Last Continuous Label ===
String lastContinuousLabel = "";

// === Setup ===
void setup()
{
    Serial.begin(115200);
    while (!Serial && millis() < 3000);

    Serial.println("Edge Impulse ML Motor Control Starting...");

    pinMode(motorA_IN1, OUTPUT);
    pinMode(motorA_IN2, OUTPUT);
    pinMode(motorB_IN3, OUTPUT);
    pinMode(motorB_IN4, OUTPUT);
    stop_all_motors();

    run_classifier_init();
    if (!microphone_inference_start(EI_CLASSIFIER_SLICE_SIZE)) {
        Serial.println("Failed to start inference!");
        while (1);
    }
}

// === Main Loop ===
void loop()
{
    unsigned long currentMillis = millis();

    // FSM for blue / high red
    if (motorState == RUNNING && currentMillis - motorStartTime >= motorRunTime) {
        stop_all_motors();
        motorStartTime = currentMillis;
        motorState = WAITING;
        lastContinuousLabel = "";  // Clear any continuous state
    }
    else if (motorState == WAITING && currentMillis - motorStartTime >= motorWaitTime) {
        motorState = IDLE;
        activeTimedLabel = "";
    }

    if (!microphone_inference_record()) return;

    signal_t signal;
    signal.total_length = EI_CLASSIFIER_SLICE_SIZE;
    signal.get_data = &microphone_audio_signal_get_data;
    ei_impulse_result_t result = {0};

    if (run_classifier_continuous(&signal, &result, debug_nn) != EI_IMPULSE_OK) {
        Serial.println("Inference error!");
        return;
    }

    // === Identify Top Classification ===
    float highest_prob = 0.0;
    const char* top_label = "background";

    Serial.println("---- Classification Scores ----");
    for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
        float val = result.classification[ix].value;
        Serial.print(result.classification[ix].label);
        Serial.print(": ");
        Serial.println(val, 4);

        if (val > highest_prob) {
            highest_prob = val;
            top_label = result.classification[ix].label;
        }
    }

    Serial.print("Top classification: ");
    Serial.print(top_label);
    Serial.print(" (");
    Serial.print(highest_prob, 4);
    Serial.println(")");

    // === Timed Motor Labels ===
    if (strcmp(top_label, "blue") == 0 && highest_prob >= THRESH_BLUE && motorState == IDLE) {
        Serial.println("Action: Motor A clockwise (timed)");
        motorA_clockwise();
        motorB_counterclockwise();
        motorStartTime = currentMillis;
        motorState = RUNNING;
        activeTimedLabel = "blue";
        lastContinuousLabel = ""; // Clear continuous state
    }
    else if (strcmp(top_label, "high red") == 0 && highest_prob >= THRESH_HIGHRED && motorState == IDLE) {
        Serial.println("Action: Motor B clockwise (timed)");
        motorA_counterclockwise();
        motorB_clockwise();
        motorStartTime = currentMillis;
        motorState = RUNNING;
        activeTimedLabel = "high red";
        lastContinuousLabel = ""; // Clear continuous state
    }

    // === Continuous Motor Labels ===
    else if (motorState == IDLE) {
        if (strcmp(top_label, "Low Red") == 0 && highest_prob >= THRESH_LOWRED) {
            if (lastContinuousLabel != "Low Red") {
                Serial.println("Action: Both motors clockwise (continuous)");
                motorA_clockwise();
                motorB_clockwise();
                lastContinuousLabel = "Low Red";
            }
        }
        else if (strcmp(top_label, "Pink") == 0 && highest_prob >= THRESH_PINK) {
            if (lastContinuousLabel != "Pink") {
                Serial.println("Action: Both motors counterclockwise (continuous)");
                motorA_counterclockwise();
                motorB_counterclockwise();
                lastContinuousLabel = "Pink";
            }
        }
        else {
            // No recognized label — stop only if no recent action
            if (lastContinuousLabel != "") {
                Serial.println("No valid continuous label, stopping motors.");
                stop_all_motors();
                lastContinuousLabel = "";
            }
        }
    }

    Serial.println("-------------------------------\n");
}

// === Motor Control ===
void motorA_clockwise() {
    digitalWrite(motorA_IN1, HIGH);
    digitalWrite(motorA_IN2, LOW);
}

void motorA_counterclockwise() {
    digitalWrite(motorA_IN1, LOW);
    digitalWrite(motorA_IN2, HIGH);
}

void motorA_stop() {
    digitalWrite(motorA_IN1, LOW);
    digitalWrite(motorA_IN2, LOW);
}

void motorB_clockwise() {
    digitalWrite(motorB_IN3, HIGH);
    digitalWrite(motorB_IN4, LOW);
}

void motorB_counterclockwise() {
    digitalWrite(motorB_IN3, LOW);
    digitalWrite(motorB_IN4, HIGH);
}

void motorB_stop() {
    digitalWrite(motorB_IN3, LOW);
    digitalWrite(motorB_IN4, LOW);
}

void stop_all_motors() {
    motorA_stop();
    motorB_stop();
}

// === Edge Impulse Audio Inference ===
static void pdm_data_ready_inference_callback(void)
{
    int bytesAvailable = PDM.available();
    int bytesRead = PDM.read((char *)&sampleBuffer[0], bytesAvailable);

    if (record_ready) {
        for (int i = 0; i < bytesRead >> 1; i++) {
            inference.buffers[inference.buf_select][inference.buf_count++] = sampleBuffer[i];
            if (inference.buf_count >= inference.n_samples) {
                inference.buf_select ^= 1;
                inference.buf_count = 0;
                inference.buf_ready = 1;
            }
        }
    }
}

static bool microphone_inference_start(uint32_t n_samples)
{
    inference.buffers[0] = (signed short *)malloc(n_samples * sizeof(signed short));
    if (!inference.buffers[0]) return false;

    inference.buffers[1] = (signed short *)malloc(n_samples * sizeof(signed short));
    if (!inference.buffers[1]) {
        free(inference.buffers[0]);
        return false;
    }

    sampleBuffer = (signed short *)malloc((n_samples >> 1) * sizeof(signed short));
    if (!sampleBuffer) {
        free(inference.buffers[0]);
        free(inference.buffers[1]);
        return false;
    }

    inference.buf_select = 0;
    inference.buf_count = 0;
    inference.n_samples = n_samples;
    inference.buf_ready = 0;

    PDM.onReceive(&pdm_data_ready_inference_callback);
    PDM.setBufferSize((n_samples >> 1) * sizeof(int16_t));
    if (!PDM.begin(1, EI_CLASSIFIER_FREQUENCY)) {
        return false;
    }

    PDM.setGain(127);
    record_ready = true;

    return true;
}

static bool microphone_inference_record(void)
{
    while (inference.buf_ready == 0) {
        delay(1);
    }

    inference.buf_ready = 0;
    return true;
}

static int microphone_audio_signal_get_data(size_t offset, size_t length, float *out_ptr)
{
    numpy::int16_to_float(&inference.buffers[inference.buf_select ^ 1][offset], out_ptr, length);
    return 0;
}

static void microphone_inference_end(void)
{
    PDM.end();
    free(inference.buffers[0]);
    free(inference.buffers[1]);
    free(sampleBuffer);
}

#if !defined(EI_CLASSIFIER_SENSOR) || EI_CLASSIFIER_SENSOR != EI_CLASSIFIER_SENSOR_MICROPHONE
#error "Invalid model for current sensor."
#endif
