/*
* Copyright (c) 2014-2021, NVIDIA CORPORATION. All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a
* copy of this software and associated documentation files (the "Software"),
* to deal in the Software without restriction, including without limitation
* the rights to use, copy, modify, merge, publish, distribute, sublicense,
* and/or sell copies of the Software, and to permit persons to whom the
* Software is furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
* THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
* FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
* DEALINGS IN THE SOFTWARE.
*/

#include <donut/engine/AudioCache.h>

#include <donut/engine/ThreadPool.h>
#include <donut/core/vfs/VFS.h>
#include <donut/core/log.h>

#include <cstdint>
#include <cstring>

using namespace donut;

// RIFF (wav) file chunk headers

struct RiffChunk
{
    char      riff[4];        // RIFF signature
    uint32_t  chunkSize;      // chunksize

    bool valid() const { return memcmp(&riff, "RIFF", 4)==0; }
};

struct WaveChunk
{
    char      wave[4],        // WAVE signature
              fmt[4];         // format header
    uint32_t  fmtChunkSize;   // Size of the fmt chunk
    uint16_t  audioFormat,    // Audio format 1=PCM,6=mulaw,7=alaw, 257=IBM Mu-Law, 258=IBM A-Law, 259=ADPCM
              numChannels;    // Number of channels 1=Mono 2=Sterio
    uint32_t  samplesPerSec,  // Sampling Frequency in Hz
              bytesPerSec;    // bytes per second
    uint16_t  blockAlign,     // 2=16-bit mono, 4=16-bit stereo
              bitsPerSample;  // Number of bits per sample

    bool valid() const { return memcmp(&wave, "WAVE", 4)==0 && memcmp(&fmt, "fmt ", 4)==0; }
};

struct DataChunk
{
    char      dataChunkID[4]; // "data"  string
    uint32_t  dataChunkSize;  // Sampled data length
};

// Declared here rather than pulling in stb_vorbis.h's full single-header
// interface (which defines its own struct types) - only the one-shot
// decode-to-malloc'd-PCM entry point is needed. Implemented by
// src/engine/stb_vorbis_impl.c (compiled as C, hence extern "C" to match
// its linkage).
extern "C" int stb_vorbis_decode_memory(const unsigned char* mem, int len, int* channels, int* sample_rate, short** output);

namespace donut::engine::audio
{

AudioCache::AudioCache(std::shared_ptr<vfs::IFileSystem> fs) : m_fs(fs) { }

void AudioCache::Reset()
{
    std::lock_guard<std::mutex> guard(m_LoadedDataMutex);

    m_LoadedAudioData.clear();
}

std::shared_ptr<AudioData const> AudioCache::importRiff(std::shared_ptr<donut::vfs::IBlob> blob, char const * filepath)
{
    uint8_t const * data = (uint8_t const *)blob->data(),
                  * ptr = data;

    RiffChunk const * riffchunk = (RiffChunk const *)ptr;
    if (!riffchunk->valid())
    {
        log::warning("Invalid RIFF header `%`", filepath);
        return nullptr;
    }
    if (riffchunk->chunkSize!=blob->size()-8) {
        log::warning("RIFF invalid chunk size `%`", filepath);
        return nullptr;
    }
    ptr += sizeof(RiffChunk);

    WaveChunk const * wavechunk = (WaveChunk const *)ptr;
    if (!wavechunk->valid())
    {
        log::warning("Invalid Wave chunk header `%s`", filepath);
        return nullptr;
    }
    if (wavechunk->fmtChunkSize<16)
    {
        log::warning("Wave chunk header invalid size `%s`", filepath);
        return nullptr;
    }
    if (wavechunk->audioFormat!=1)
    {
        log::warning("Wave chunk header unsupported format %d (PCM=1) `%s`", wavechunk->audioFormat, filepath);
        return nullptr;
    }
    ptr += sizeof(WaveChunk);

    DataChunk const * datachunk = nullptr;
    for ( ; ptr < data+(blob->size()-4); ++ptr)
        if (memcmp(ptr, "data", 4)==0)
        {
            datachunk = (DataChunk const *)ptr;
            break;
        }

    if (!datachunk)
    {
        log::warning("Cannot find Data chunk `%s`", filepath);
        return nullptr;
    }
    if (ptr+datachunk->dataChunkSize>=data+blob->size())
    {
        log::warning("Invalid data chunk size `%s`", filepath);
        return nullptr;
    }
    ptr += sizeof(DataChunk);

    std::shared_ptr<AudioData> result = std::make_shared<AudioData>();

    result->format = AudioData::Format::WAVE_PCM_INTEGER;
    result->nchannels = wavechunk->numChannels;
    result->sampleRate = wavechunk->samplesPerSec;
    result->byteRate = wavechunk->bytesPerSec;
    result->bitsPerSample = wavechunk->bitsPerSample;
    result->blockAlignment = wavechunk->blockAlign;

    result->samplesSize = datachunk->dataChunkSize;
    result->samples = ptr;

    result->m_data = blob;

    return result;
}

std::shared_ptr<AudioData const> AudioCache::importOgg(std::shared_ptr<donut::vfs::IBlob> blob, char const * filepath)
{
    int channels = 0, sampleRate = 0;
    short* pcm = nullptr;
    int frames = stb_vorbis_decode_memory((unsigned char const*)blob->data(), int(blob->size()), &channels, &sampleRate, &pcm);
    if (frames <= 0 || !pcm)
    {
        log::warning("Failed to decode Ogg Vorbis file `%s`", filepath);
        return nullptr;
    }

    std::shared_ptr<AudioData> result = std::make_shared<AudioData>();

    result->format = AudioData::Format::WAVE_PCM_INTEGER;
    result->nchannels = uint32_t(channels);
    result->sampleRate = uint32_t(sampleRate);
    result->bitsPerSample = 16;
    result->blockAlignment = uint16_t(channels * (result->bitsPerSample / 8));
    result->byteRate = result->sampleRate * result->blockAlignment;

    size_t dataSize = size_t(frames) * size_t(channels) * sizeof(short);
    result->samplesSize = uint32_t(dataSize);
    result->samples = pcm;

    // stb_vorbis_decode_memory malloc()s `pcm` - vfs::Blob's destructor
    // calls free() on the pointer it's given, exactly matching (see
    // VFS.cpp Blob::~Blob).
    result->m_data = std::make_shared<donut::vfs::Blob>(pcm, dataSize);

    return result;
}

static bool strcaseequals(const std::string& a, const std::string& b)
{
#ifdef _WIN32
    return _stricmp(a.c_str(), b.c_str()) == 0;
#else
    return strcasecmp(a.c_str(), b.c_str()) == 0;
#endif
}

std::shared_ptr<AudioData const> AudioCache::loadAudioFile (const std::filesystem::path & path)
{

    std::shared_ptr<vfs::IBlob> blob = m_fs->readFile(path);
    if (!blob)
    {
        log::warning("Couldn't read audio file `%s`", path.generic_string().c_str());
        return nullptr;
    }

    auto extension = path.extension();
    if (strcaseequals(extension.generic_string(), ".wav"))
    {
        return importRiff(blob, path.generic_string().c_str());
    }
    else if (strcaseequals(extension.generic_string(), ".ogg"))
    {
        return importOgg(blob, path.generic_string().c_str());
    }
    else
        log::warning("Unsupported audio format `%s` for file `%s`", extension.c_str());

    return nullptr;
}

bool AudioCache::findInCache(const std::filesystem::path & path, std::shared_ptr<AudioData const> & result)
{
    result.reset();

    std::lock_guard<std::mutex> guard(m_LoadedDataMutex);

    result = m_LoadedAudioData[path.generic_string()];
    if (result)
        return true;

    result = std::make_shared<AudioData const>();
    m_LoadedAudioData[path.generic_string()] = result;
    return false;
}

void AudioCache::sendAudioLoadedMessage(std::shared_ptr<AudioData const> audio, char const * path)
{
    log::info("Loaded (%dkHz) : %s", audio->sampleRate/1000, path);
}

std::shared_ptr<AudioData const> AudioCache::LoadFromFile(const std::filesystem::path & path)
{
    std::shared_ptr<AudioData const> audio;

    if (findInCache(path, audio))
        return audio;

    if ((audio = loadAudioFile (path)))
    {
        sendAudioLoadedMessage(audio, path.generic_string().c_str());
    }
    return audio;
}

std::shared_ptr<AudioData const> AudioCache::LoadFromFileAsync(const std::filesystem::path & path, ThreadPool& threadPool)
{
    std::shared_ptr<AudioData const> audio;

    if (findInCache(path, audio))
        return audio;

    threadPool.AddTask([this, &audio, path]()
    {
        if ((audio = loadAudioFile(path)))
        {
            sendAudioLoadedMessage(audio, path.generic_string().c_str());
        }
    });
    return audio;
}

} // namespace donut::engine::audio
