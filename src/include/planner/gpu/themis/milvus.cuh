#ifndef HTEMIS_MILVUS_H
#define HTEMIS_MILVUS_H

#include <iostream>
#include <string>
#include "./httplib.h"
#include <nlohmann/json.hpp>
#include <chrono>

using json = nlohmann::json;

void listCollections(std::string &host) {
    httplib::Client cli(host);
    
    httplib::Headers headers = {
        {"accept", "application/json"},
        {"content-type", "application/json"}
    };

    std::string body = "{}";

    auto res = cli.Post("/v2/vectordb/collections/list", headers, body, "application/json");

    if (res && res->status == 200) {
        std::cout << "Collections: " << res->body << std::endl;
    } else {
        std::cerr << "Failed to list collections!" << std::endl;
        if (res) {
            std::cerr << "Status: " << res->status << std::endl;
        }
    }
}


void query(std::string &host, const std::string &collection_name, const std::vector<float> &query_vector, const std::string& filter, int top_k) {
    httplib::Client cli(host);

    httplib::Headers headers = {
        {"accept", "application/json"},
        {"content-type", "application/json"},
    };

    // Convert vector<float> to string
    std::string vector_str = "[";
    for (size_t i = 0; i < query_vector.size(); i++) {
        vector_str += std::to_string(query_vector[i]);
        if (i < query_vector.size() - 1) vector_str += ",";
    }
    vector_str += "]";

    std::string body = "{\n"
        "    \"collectionName\": \"" + collection_name + "\",\n"
        "    \"data\": [" + vector_str + "],\n"
        "    \"annsField\": \"embedding\",\n" 
        "    \"limit\": " + std::to_string(top_k) + ",\n"
        "    \"outputFields\": [\"id\"]\n";
    if (!filter.empty()) {
        body += ",    \"filter\": \"" + filter + "\"\n";
    }
    body += "}";


    auto res = cli.Post("/v2/vectordb/entities/search", headers, body, "application/json");

    if (res && res->status == 200) {
        // Parse JSON response
        nlohmann::json response = nlohmann::json::parse(res->body);
        
        std::vector<float> distances;
        std::vector<int> ids;
        
        // Extract distances and ids from results
        for (const auto& result : response["data"]) {
            distances.push_back(result["distance"].get<float>());
            ids.push_back(result["id"].get<int>());
        }

        // Print results
        std::cout << "Search results:" << std::endl;
        for (size_t i = 0; i < distances.size(); i++) {
            std::cout << "Distance: " << distances[i] << ", ID: " << ids[i] << std::endl;
        }
    } else {
        std::cerr << "Search failed!" << std::endl;
        if (res) {
            std::cerr << "Status: " << res->status << std::endl;
        }
    }
    
}

void query(std::string &host, const std::string &collection_name, const std::vector<std::vector<float>> &query_vectors, const std::string& filter, int top_k) {
httplib::Client cli(host);

    httplib::Headers headers = {
        {"accept", "application/json"},
        {"content-type", "application/json"},
    };

    // Convert vector<float> to string
    std::string vector_str = "[";
    for (const auto& query_vector : query_vectors) {
        vector_str += "[";
        for (size_t i = 0; i < query_vector.size(); i++) {
            // Use fixed precision to avoid scientific notation
            std::ostringstream ss;
            ss << std::fixed << std::setprecision(6) << query_vector[i];
            vector_str += ss.str();
            if (i < query_vector.size() - 1) vector_str += ",";
        }
        vector_str += "]";
        if (&query_vector != &query_vectors.back()) vector_str += ",";
    }
    vector_str += "]";

    std::string body = "{\n"
        "    \"collectionName\": \"" + collection_name + "\",\n"
        "    \"data\": " + vector_str + ",\n"
        "    \"annsField\": \"embedding\",\n" 
        "    \"limit\": " + std::to_string(top_k) + ",\n"
        "    \"outputFields\": [\"id\"]\n";
    if (!filter.empty()) {
        body += ",    \"filter\": \"" + filter + "\"\n";
    }
    body += "}";

    //std::cout << body << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    auto res = cli.Post("/v2/vectordb/entities/search", headers, body, "application/json");
    auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Query time: " << duration.count() << " milliseconds" << std::endl;

    if (res && res->status == 200) {
        // Parse JSON response
        nlohmann::json response = nlohmann::json::parse(res->body);

        std::cout << "Response: " << response << std::endl;
        
        std::vector<float> distances;
        std::vector<int> ids;
        
        // Extract distances and ids from results
        for (const auto& result : response["data"]) {
            distances.push_back(result["distance"].get<float>());
            ids.push_back(result["id"].get<int>());
        }

        // Print results
        // std::cout << "Search results:" << std::endl;
        // for (size_t i = 0; i < distances.size(); i++) {
        //     std::cout << "Distance: " << distances[i] << ", ID: " << ids[i] << std::endl;
        // }
    } else {
        std::cerr << "Search failed!" << std::endl;
        if (res) {
            std::cerr << "Status: " << res->status << std::endl;
        }
    }
}


void query(std::vector<int>& ids, std::vector<float>& distances, std::vector<int>& ranks, const std::string &host, const std::string &collection_name, float* data, int dim, int num_queries, const std::string& filter, int top_k) {
    
    httplib::Client cli(host);

    httplib::Headers headers = {
        {"accept", "application/json"},
        {"content-type", "application/json"},
    };

    // Convert vector<float> to string
    std::string vector_str = "[";
    for (int i = 0; i < num_queries; i++) {
        vector_str += "[";
        for (size_t j = 0; j < dim; j++) {
            // Use fixed precision to avoid scientific notation
            std::ostringstream ss;
            ss << std::fixed << std::setprecision(6) << data[i * dim + j];
            vector_str += ss.str();
            if (j < dim - 1) vector_str += ",";
        }
        vector_str += "]";
        if (i < num_queries - 1) vector_str += ",";
    }
    vector_str += "]";

    std::string body = "{\n"
        "    \"collectionName\": \"" + collection_name + "\",\n"
        "    \"data\": " + vector_str + ",\n"
        "    \"annsField\": \"embedding\",\n" 
        "    \"limit\": " + std::to_string(top_k) + ",\n"
        "    \"outputFields\": [\"id\"]\n";
    if (!filter.empty()) {
        body += ",    \"filter\": \"" + filter + "\"\n";
    }
    body += "}";

    //std::cout << body << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    auto res = cli.Post("/v2/vectordb/entities/search", headers, body, "application/json");
    auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Query time: " << duration.count() << " milliseconds" << std::endl;

    if (res && res->status == 200) {
        // Parse JSON response
        nlohmann::json response = nlohmann::json::parse(res->body);

        // std::cout << "Response: " << response << std::endl;

        ids.resize(0);
        distances.resize(0);
        ranks.resize(0);


        // Extract distances and ids from results
        int i = 0;
        for (const auto& result : response["data"]) {
            distances.push_back(result["distance"].get<float>());
            ids.push_back(result["id"].get<int>());
            ranks.push_back(i % top_k);
            i++;
        }

        // Print results
        // std::cout << "Search results:" << std::endl;
        // for (size_t i = 0; i < distances.size(); i++) {
        //     std::cout << "Distance: " << distances[i] << ", ID: " << ids[i] << std::endl;
        // }
    } else {
        std::cerr << "Search failed!" << std::endl;
        if (res) {
            std::cerr << "Status: " << res->status << std::endl;
        }
    }
}

void query(std::vector<int>& ids, std::vector<float>& distances, std::vector<int>& ranks, const std::string &host, const std::string &collection_name, float* data, int* indices, int dim, int num_queries, const std::string& filter, int top_k) {
    
    httplib::Client cli(host);

    httplib::Headers headers = {
        {"accept", "application/json"},
        {"content-type", "application/json"},
    };

    // Convert vector<float> to string
    std::string vector_str = "[";
    for (int i = 0; i < num_queries; i++) {
        vector_str += "[";
        for (size_t j = 0; j < dim; j++) {
            // Use fixed precision to avoid scientific notation
            std::ostringstream ss;
            ss << std::fixed << std::setprecision(6) << data[indices[i] * dim + j];
            vector_str += ss.str();
            if (j < dim - 1) vector_str += ",";
        }
        vector_str += "]";
        if (i < num_queries - 1) vector_str += ",";
    }
    vector_str += "]";

    std::string body = "{\n"
        "    \"collectionName\": \"" + collection_name + "\",\n"
        "    \"data\": " + vector_str + ",\n"
        "    \"annsField\": \"embedding\",\n" 
        "    \"limit\": " + std::to_string(top_k) + ",\n"
        "    \"outputFields\": [\"id\"]\n";
    if (!filter.empty()) {
        body += ",    \"filter\": \"" + filter + "\"\n";
    }
    body += "}";

    //std::cout << body << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    auto res = cli.Post("/v2/vectordb/entities/search", headers, body, "application/json");
    auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Query time: " << duration.count() << " milliseconds" << std::endl;

    if (res && res->status == 200) {
        // Parse JSON response
        nlohmann::json response = nlohmann::json::parse(res->body);

        // std::cout << "Response: " << response << std::endl;

        ids.resize(0);
        distances.resize(0);
        ranks.resize(0);


        // Extract distances and ids from results
        int i = 0;
        for (const auto& result : response["data"]) {
            distances.push_back(result["distance"].get<float>());
            ids.push_back(result["id"].get<int>());
            ranks.push_back(i % top_k);
            i++;
        }

        // Print results
        // std::cout << "Search results:" << std::endl;
        // for (size_t i = 0; i < distances.size(); i++) {
        //     std::cout << "Distance: " << distances[i] << ", ID: " << ids[i] << std::endl;
        // }
    } else {
        std::cerr << "Search failed!" << std::endl;
        if (res) {
            std::cerr << "Status: " << res->status << std::endl;
        }
    }
}

__global__ void setMilvusResultOffsets(int* offsets, int num_queries, int top_k) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = tid; i < num_queries; i += blockDim.x * gridDim.x) {
        offsets[2*i] = i * top_k;
        offsets[2*i+1] = (i+1) * top_k;
    }
}


__global__ void cagra_gather_queries(float* queries, int* indices, float* data, int num_queries, int dim) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    
    // Each thread handles multiple dimensions in parallel
    for (int i = tid; i < num_queries * dim; i += stride) {
        int query_idx = i / dim;
        int dim_idx = i % dim;
        if (query_idx < num_queries) {
            queries[i] = data[indices[query_idx] * dim + dim_idx];
        }
    }
}



#endif