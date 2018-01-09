/**
 * Copyright (c) 2017 Melown Technologies SE
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * *  Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * *  Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef vts_tools_tmptsencoder_hpp_included_
#define vts_tools_tmptsencoder_hpp_included_

#include <boost/optional.hpp>

#include "utility/gccversion.hpp"

#include "vts-libs/vts/encoder.hpp"
#include "vts-libs/tools-support/progress.hpp"
#include "vts-libs/vts/ntgenerator.hpp"

#include "./tmptileset.hpp"

namespace vtslibs { namespace vts { namespace tools {

class TmpTsEncoder : public Encoder {
public:
    struct Config {
        storage::CreditIds credits;
        int textureQuality;
        double dtmExtractionRadius;

        bool forceWatertight;
        bool resume;
        bool keepTmpset;

        bool fuseSubmeshes;
        SubmeshMergeOptions smMergeOptions;

        boost::optional<TileId> debug_tileId;

        Config()
            : textureQuality(85), dtmExtractionRadius(40.0)
            , forceWatertight(false), resume(false), keepTmpset(false)
            , fuseSubmeshes(true)
        {}

        void configuration(boost::program_options::options_description
                           &config);
        void configure(const boost::program_options::variables_map &vars);
    };

    typedef vtslibs::tools::ExternalProgress ExternalProgress;

    TmpTsEncoder(const boost::filesystem::path &path
                 , const TileSetProperties &properties
                 , CreateMode mode
                 , const Config &config
                 , ExternalProgress::Config &&epConfig
                 , const ExternalProgress::Weights &weights);

    ~TmpTsEncoder();

    void run();

protected:
    vtslibs::tools::ExternalProgress& progress() { return progress_; }

    TmpTileset& tmpset() { return tmpset_; }

    NtGenerator& ntg() { return ntg_; }

private:
    void prepare();

    virtual TileResult
    generate(const TileId &tileId, const NodeInfo &nodeInfo
             , const TileResult&) UTILITY_OVERRIDE;

    virtual void finish(TileSet &ts);

    const Config config_;

    vtslibs::tools::ExternalProgress progress_;

    TmpTileset tmpset_;
    TileIndex index_;
    TileIndex deriveTree_;
    TileIndex validTree_;

    NtGenerator ntg_;
};

} } } // namespace vtslibs::tools

#endif // vts_tools_tmptsencoder_hpp_included_
