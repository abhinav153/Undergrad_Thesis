from pylsl import StreamInlet,resolve_stream

print("looking for an EMG stream...")
streams = resolve_stream()

# create a new inlet to read from the stream
inlet = StreamInlet(streams[0])
print(inlet.info().type())


# get a new sample (you can also omit the timestamp part if you're not
# interested in it)
sample, timestamp = inlet.pull_sample()
print(timestamp, sample)