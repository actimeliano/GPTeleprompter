class AudioProcessor extends AudioWorkletProcessor {
  process(inputs, outputs, parameters) {
    const input = inputs[0];
    const audioData = input[0];
    
    if (audioData) {
      this.port.postMessage({ audioData: audioData });
    }
    
    return true;
  }
}

registerProcessor('audio-processor', AudioProcessor);