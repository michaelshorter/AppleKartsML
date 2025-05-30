#define EIDSP_QUANTIZE_FILTERBANK   0
#define EI_CLASSIFIER_SLICES_PER_MODEL_WINDOW 4

#include <PDM.h>
#include <AppleKarts_ML_inferencing.h>

// LED pin definitions
const int ledLowRed = 2;
const int ledPink = 3;
const int ledBlue = 4;
const int ledHighRed = 5;

/** Audio buffers, pointers and selectors */
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

void setup()
{
    // LED setup
    pinMode(ledLowRed, OUTPUT);
    pinMode(ledPink, OUTPUT);
    pinMode(ledBlue, OUTPUT);
    pinMode(ledHighRed, OUTPUT);
    digitalWrite(ledLowRed, LOW);
    digitalWrite(ledPink, LOW);
    digitalWrite(ledBlue, LOW);
    digitalWrite(ledHighRed, LOW);

    run_classifier_init();
    if (!microphone_inference_start(EI_CLASSIFIER_SLICE_SIZE)) {
        while (1); // Stop if memory allocation failed
    }
}

void loop()
{
    if (!microphone_inference_record()) {
        return;
    }

    signal_t signal;
    signal.total_length = EI_CLASSIFIER_SLICE_SIZE;
    signal.get_data = &microphone_audio_signal_get_data;
    ei_impulse_result_t result = {0};

    if (run_classifier_continuous(&signal, &result, debug_nn) != EI_IMPULSE_OK) {
        return;
    }

    // Find top class from current slice
    float highest_prob = 0.0;
    const char* top_label = "background";

    for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
        float val = result.classification[ix].value;
        if (val > highest_prob) {
            highest_prob = val;
            top_label = result.classification[ix].label;
        }
    }

    // Reset all LEDs
    digitalWrite(ledLowRed, LOW);
    digitalWrite(ledPink, LOW);
    digitalWrite(ledBlue, LOW);
    digitalWrite(ledHighRed, LOW);

    // Turn on only one LED based on class
    if (strcmp(top_label, "Low Red") == 0) {
        digitalWrite(ledLowRed, HIGH);
    }
    else if (strcmp(top_label, "Pink") == 0) {
        digitalWrite(ledPink, HIGH);
    }
    else if (strcmp(top_label, "blue") == 0) {
        digitalWrite(ledBlue, HIGH);
    }
    else if (strcmp(top_label, "high red") == 0) {
        digitalWrite(ledHighRed, HIGH);
    }
}

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
    if (inference.buf_ready == 1) {
        return false;
    }

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
