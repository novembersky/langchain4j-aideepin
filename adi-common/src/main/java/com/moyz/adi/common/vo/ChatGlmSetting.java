package com.moyz.adi.common.vo;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;

@Data
public class ChatGlmSetting extends CommonAiPlatformSetting {

    @JsonProperty("api_key")
    private String apiKey;

}
