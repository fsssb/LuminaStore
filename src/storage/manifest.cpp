#include "lumina/storage/manifest.h"

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

namespace lumina {
namespace {

constexpr size_t kManifestKeep = 2;

Status io_error(const std::string& what) {
    return Status::IOError(what + " (" + std::strerror(errno) + ")");
}

std::string fsync_dir_path(const std::string& file_path) {
    const size_t slash = file_path.find_last_of('/');
    if (slash == std::string::npos) {
        return ".";
    }
    return file_path.substr(0, slash);
}

bool parse_line(const std::string& line, ManifestEntry* out) {
    std::istringstream is(line);
    std::string tag;
    uint64_t seq = 0, off = 0;
    std::string filename;
    if (!(is >> tag >> seq >> off >> filename)) {
        return false;
    }
    if (tag != "snap") {
        return false;
    }
    out->seq = seq;
    out->wal_offset = off;
    out->filename = std::move(filename);
    return true;
}

}  // namespace

Status read_manifest_latest(const std::string& manifest_path, ManifestEntry* out) {
    if (out == nullptr) {
        return Status::InvalidArgument("out is null");
    }
    std::ifstream ifs(manifest_path);
    if (!ifs) {
        return Status::NotFound("manifest missing: " + manifest_path);
    }

    ManifestEntry latest;
    bool found = false;
    std::string line;
    while (std::getline(ifs, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        ManifestEntry e;
        if (!parse_line(line, &e)) {
            return Status::Corruption("unparseable manifest line: " + line);
        }
        latest = std::move(e);
        found = true;
    }
    if (!found) {
        return Status::NotFound("manifest empty");
    }
    *out = std::move(latest);
    return Status::OK();
}

Status write_manifest_append(const std::string& manifest_path, const ManifestEntry& entry) {
    // Read existing records (keep the most recent kManifestKeep).
    std::vector<std::string> kept;
    {
        std::ifstream ifs(manifest_path);
        if (ifs) {
            std::vector<std::string> all;
            std::string line;
            while (std::getline(ifs, line)) {
                if (!line.empty() && line[0] != '#') {
                    all.push_back(line);
                }
            }
            if (all.size() > kManifestKeep) {
                all.erase(all.begin(), all.begin() + static_cast<long>(all.size() - kManifestKeep));
            }
            kept = std::move(all);
        }
    }

    const std::string tmp_path = manifest_path + ".tmp";
    const std::string content = "snap " + std::to_string(entry.seq) + " " +
                                std::to_string(entry.wal_offset) + " " + entry.filename + "\n";

    const int fd = ::open(tmp_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        return io_error("open manifest tmp");
    }

    bool ok = true;
    std::string full;
    for (const auto& l : kept) {
        full += l;
        full += '\n';
    }
    full += content;

    size_t written = 0;
    while (written < full.size()) {
        const ssize_t n = ::write(fd, full.data() + written, full.size() - written);
        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            ok = false;
            break;
        }
        if (n == 0) {
            ok = false;
            break;
        }
        written += static_cast<size_t>(n);
    }
    if (ok && ::fsync(fd) != 0) {
        ok = false;
    }
    ::close(fd);
    if (!ok) {
        ::unlink(tmp_path.c_str());
        return io_error("write manifest tmp");
    }
    if (::rename(tmp_path.c_str(), manifest_path.c_str()) != 0) {
        ::unlink(tmp_path.c_str());
        return io_error("rename manifest");
    }

    // Best-effort dir fsync for durability of the rename.
    const std::string dir = fsync_dir_path(manifest_path);
    const int dfd = ::open(dir.c_str(), O_RDONLY | O_DIRECTORY);
    if (dfd >= 0) {
        ::fsync(dfd);
        ::close(dfd);
    }
    return Status::OK();
}

} // namespace lumina
