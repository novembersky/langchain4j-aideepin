package com.moyz.adi.common.service;

import com.moyz.adi.common.cosntant.AdiConstant;
import com.moyz.adi.common.interfaces.AbstractLLMService;
import com.moyz.adi.common.vo.ChatGlmSetting;
import com.moyz.adi.common.vo.QianFanSetting;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.chat.StreamingChatLanguageModel;
import dev.langchain4j.model.chatglm.ChatGlmChatModel;
import dev.langchain4j.model.qianfan.QianfanChatModel;
import dev.langchain4j.model.qianfan.QianfanStreamingChatModel;
import dev.langchain4j.model.zhipu.ZhipuAiStreamingChatModel;
import lombok.experimental.Accessors;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;

import java.time.Duration;

/**
 * ChatGlm LLM service
 */
@Slf4j
@Accessors(chain = true)
public class ChatGlmLLMService extends AbstractLLMService<ChatGlmSetting> {

    public ChatGlmLLMService(String modelName) {
        super(modelName, AdiConstant.SysConfigKey.CHATGLM_SETTING, ChatGlmSetting.class);
    }

    @Override
    public boolean isEnabled() {
        return StringUtils.isNoneBlank(setting.getApiKey());
    }

    @Override
    protected ChatLanguageModel buildChatLLM() {

        return ChatGlmChatModel.builder().baseUrl("https://open.bigmodel.cn/api/paas/v4/chat/completions/")
                .topP(1.0).maxRetries(1).temperature(0.7).maxLength(3000)
                .timeout(Duration.ofSeconds(20L)).build();

    }

    @Override
    protected StreamingChatLanguageModel buildStreamingChatLLM() {

        return ZhipuAiStreamingChatModel.builder().apiKey(setting.getApiKey())
                .baseUrl("https://open.bigmodel.cn/")
                .temperature(0.7).model("glm-3-turbo")
                .build();

    }

    @Override
    protected String parseError(Object error) {
        return null;
    }
}
