package org.dbiir.txnagent.execution.isolation;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.dbiir.txnagent.common.types.IsolationLevelType;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.yaml.snakeyaml.Yaml;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class PartitionConfig {
  private static final Logger logger = LoggerFactory.getLogger(PartitionConfig.class);
  private String name;
  private Integer scalaFactor;
  private Integer partitionCount;
  private List<Relation> relations;
  private List<Partition> partitions;
  private List<Stage> stages;

  @SuppressWarnings("unchecked")
  public void loadFromMap(Map<String, Object> data) {
    this.name = (String) data.get("name");
    this.scalaFactor = (Integer) data.get("scalaFactor");
    this.partitionCount = (Integer) data.get("partitionCount");

    // convert relations
    List<Map<String, Object>> relationData = (List<Map<String, Object>>) data.get("relations");
    if (relationData != null) {
      List<Relation> relations = new ArrayList<>();

      for (Map<String, Object> relationMap : relationData) {
        Relation relation = new Relation();
        relation.setName((String) relationMap.get("name"));
        relation.setPartitionSize((Integer) relationMap.get("partitionSize"));
        relation.setPartitionCount((Integer) relationMap.get("partitionCount"));
        relations.add(relation);
      }

      this.relations = relations;
    }

    // convert partitions
    List<Map<String, Object>> partitionData = (List<Map<String, Object>>) data.get("partitions");
    if (partitionData != null) {
      List<Partition> partitions = new ArrayList<>();
      for (Map<String, Object> partitionMap : partitionData) {
        Partition partition = new Partition();
        partition.setId((Integer) partitionMap.get("id"));
        partition.setRelationName((String) partitionMap.get("relationName"));
        partitions.add(partition);
      }
      this.partitions = partitions;
    }

    // convert stages
    List<Map<String, Object>> stageData = (List<Map<String, Object>>) data.get("stages");
    if (stageData != null) {
      List<Stage> stages = new ArrayList<>();

      for (Map<String, Object> stageMap : stageData) {
        Stage stage = new Stage();
        stage.setId((Integer) stageMap.get("id"));
        stage.setStartTime((Integer) stageMap.get("startTime"));
        stage.setEndTime((Integer) stageMap.get("endTime"));

        // convert partition
        List<Map<String, Object>> isolationData =
            (List<Map<String, Object>>) stageMap.get("isolation");
        if (isolationData != null) {
          List<IsolationConfig> isolations = new ArrayList<>();

          for (Map<String, Object> isolationMap : isolationData) {
            IsolationConfig isolation = new IsolationConfig();
            isolation.setId((Integer) isolationMap.get("id"));

            // convert isolation
            String isolationStr = (String) isolationMap.get("level");
            if (isolationStr != null) {
              try {
                isolation.setLevel(IsolationLevelType.valueOf(isolationStr));
              } catch (IllegalArgumentException e) {
                logger.error("Unknown isolation level: {}, using default", isolationStr);
                isolation.setLevel(IsolationLevelType.SER);
              }
            }
            isolations.add(isolation);
          }

          stage.setIsolations(isolations);
        }

        stages.add(stage);
      }

      this.stages = stages;
    }
  }

  public PartitionConfig load(String filePath) throws IOException {
    // 1. read yaml file
    Map<String, Object> yamlData = readYamlFile(filePath);

    // 2. extract workload data
    Map<String, Object> workloadData = extractWorkloadData(yamlData);

    // 3. create and load
    this.loadFromMap(workloadData);

    return this;
  }

  private Map<String, Object> readYamlFile(String filePath) throws IOException {
    Yaml yaml = new Yaml();
    try (InputStream inputStream = Files.newInputStream(Paths.get(filePath))) {
      return yaml.load(inputStream);
    }
  }

  @SuppressWarnings("unchecked")
  private Map<String, Object> extractWorkloadData(Map<String, Object> yamlData) {
    if (yamlData == null) {
      throw new IllegalArgumentException("YAML data cannot be null");
    }

    if (yamlData.containsKey("workload")) {
      return (Map<String, Object>) yamlData.get("workload");
    }

    return yamlData;
  }

  public Stage getStageById(int stageId) {
    return stages.stream().filter(s -> s.getId() == stageId).findFirst().orElse(null);
  }

  public Stage getStageByTime(long currentTime) {
    return stages.stream()
        .filter(s -> currentTime >= s.getStartTime() && currentTime < s.getEndTime())
        .findFirst()
        .orElse(null);
  }

  public IsolationLevelType getIsolationLevel(long currentTime, int partitionId) {
    Stage stageConfig = getStageByTime(currentTime);
    if (stageConfig != null) {
      return stageConfig.getIsolationForPartition(partitionId);
    }
    return IsolationLevelType.SER;
  }

  @Data
  @NoArgsConstructor
  @AllArgsConstructor
  public static class Relation {
    private String name;
    private Integer partitionSize; // partitionSize in YAML
    private Integer partitionCount;
  }

  @Data
  @NoArgsConstructor
  @AllArgsConstructor
  public static class Partition {
    private Integer id;
    private String relationName;
  }

  @Data
  @NoArgsConstructor
  @AllArgsConstructor
  public static class IsolationConfig {
    private Integer id;
    private IsolationLevelType level; // level in YAML
  }

  @Data
  @NoArgsConstructor
  @AllArgsConstructor
  public static class Stage {
    private Integer id;
    private Integer startTime;
    private Integer endTime;
    private List<IsolationConfig> isolations;

    public IsolationLevelType getIsolationForPartition(int partitionId) {
      if (isolations == null) return null;
      return isolations.stream()
          .filter(i -> i.getId() == partitionId)
          .findFirst()
          .map(IsolationConfig::getLevel)
          .orElse(null);
    }
  }
}
