package org.dbiir.txnagent.common;

import com.fasterxml.jackson.dataformat.xml.annotation.JacksonXmlProperty;
import com.fasterxml.jackson.dataformat.xml.annotation.JacksonXmlRootElement;
import lombok.Getter;
import lombok.Setter;

@Setter
@Getter
@JacksonXmlRootElement(localName = "parameters")
public class JacksonXmlConfiguration {

  // Getters and setters
  @JacksonXmlProperty(localName = "type")
  private String type;

  @JacksonXmlProperty(localName = "driver")
  private String driver;

  @JacksonXmlProperty(localName = "url")
  private String url;

  @JacksonXmlProperty(localName = "username")
  private String username;

  @JacksonXmlProperty(localName = "password")
  private String password;

  @JacksonXmlProperty(localName = "reconnectOnConnectionFailure")
  private boolean reconnectOnConnectionFailure;

  @JacksonXmlProperty(localName = "newConnectionPerTxn")
  private boolean newConnectionPerTxn;

  @JacksonXmlProperty(localName = "isolation")
  private String isolation;

  @JacksonXmlProperty(localName = "terminals")
  private int terminals;

  @JacksonXmlProperty(localName = "batchsize")
  private int batchsize;

  @JacksonXmlProperty(localName = "concurrencyControlType")
  private String concurrencyControlType;

  @JacksonXmlProperty(localName = "benchmark")
  private String benchmark;

  @JacksonXmlProperty(localName = "randomSeed")
  private int randomSeed;

  @JacksonXmlProperty(localName = "maxRetries")
  private int maxRetries;

  @JacksonXmlProperty(localName = "scalefactor")
  private double scalefactor;

  // Initialize values
  public JacksonXmlConfiguration() {
    this.type = "postgresql";
    this.driver = "org.postgresql.Driver";
    this.url = "jdbc:postgresql://localhost:5432/osprey";
    this.username = "postgres";
    this.password = "Ss123!@#";
    this.reconnectOnConnectionFailure = true;
    this.newConnectionPerTxn = false;
    this.isolation = "SERIALIZABLE";
    this.terminals = 128;
    this.batchsize = 128;
    this.concurrencyControlType = "SI_TAILOR";
    this.benchmark = "smallbank";
    this.randomSeed = 1;
    this.maxRetries = 3;
    this.scalefactor = 1.0;
  }
}
