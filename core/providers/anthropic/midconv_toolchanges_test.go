package anthropic

import (
	"strings"
	"testing"

	providerUtils "github.com/maximhq/bifrost/core/providers/utils"
	"github.com/maximhq/bifrost/core/schemas"
)

// setMidConvSystemRecord installs a datasheet record saying (provider, model) accepts
// mid-conversation system messages, as the datasheet did for vertex_ai Opus 5.5.
func setMidConvSystemRecord(t *testing.T, provider schemas.ModelProvider) {
	t.Helper()
	yes := true
	providerUtils.SetCapabilityResolver(func(p schemas.ModelProvider, m string) *schemas.ModelCapabilities {
		if p == provider && strings.Contains(m, "claude-opus-5-5") {
			return &schemas.ModelCapabilities{SupportsMidConversationSystem: &yes}
		}
		return nil
	})
	t.Cleanup(func() { providerUtils.SetCapabilityResolver(nil) })
}

// toolAdditionRequest: a tool loop whose last item is a system message offering a deferred
// tool, optionally with a text block beside it (falcon's deferred tool loading shape).
func toolAdditionRequest(provider schemas.ModelProvider, withText bool) *schemas.BifrostResponsesRequest {
	blocks := []schemas.ResponsesMessageContentBlock{{
		Type: schemas.ResponsesInputMessageContentBlockTypeToolAddition,
		ToolChange: &schemas.ResponsesToolChangeTarget{
			Type: schemas.ResponsesToolChangeTargetTypeToolReference,
			Name: schemas.Ptr("dhcp_leases"),
		},
	}}
	if withText {
		blocks = append(blocks, schemas.ResponsesMessageContentBlock{
			Type: schemas.ResponsesInputMessageContentBlockTypeText,
			Text: schemas.Ptr("dhcp_leases is now available"),
		})
	}
	return &schemas.BifrostResponsesRequest{
		Provider: provider,
		Model:    "claude-opus-5-5",
		Input: []schemas.ResponsesMessage{
			{Role: schemas.Ptr(schemas.ResponsesInputMessageRoleUser), Content: &schemas.ResponsesMessageContent{ContentStr: schemas.Ptr("find the IP for this MAC")}},
			{
				Type:                 schemas.Ptr(schemas.ResponsesMessageTypeFunctionCall),
				ResponsesToolMessage: &schemas.ResponsesToolMessage{CallID: schemas.Ptr("toolu_1"), Name: schemas.Ptr("tool_search"), Arguments: schemas.Ptr(`{"query":"dhcp"}`)},
				Status:               schemas.Ptr("completed"),
			},
			{
				Type:                 schemas.Ptr(schemas.ResponsesMessageTypeFunctionCallOutput),
				ResponsesToolMessage: &schemas.ResponsesToolMessage{CallID: schemas.Ptr("toolu_1"), Output: &schemas.ResponsesToolMessageOutputStruct{ResponsesToolCallOutputStr: schemas.Ptr("dhcp_leases")}},
			},
			{Role: schemas.Ptr(schemas.ResponsesInputMessageRoleSystem), Content: &schemas.ResponsesMessageContent{ContentBlocks: blocks}},
		},
		Params: &schemas.ResponsesParameters{MaxOutputTokens: schemas.Ptr(256)},
	}
}

func countToolChangeBlocks(msgs []AnthropicMessage) (n int, systemText int) {
	for _, m := range msgs {
		for _, b := range m.Content.ContentBlocks {
			if isToolChangeBlock(b.Type) {
				n++
			}
			if m.Role == AnthropicMessageRoleSystem && b.Type == AnthropicContentBlockTypeText {
				systemText++
			}
		}
	}
	return n, systemText
}

// A provider without MidConvToolChanges never gets the beta, so a datasheet row enabling
// mid-conversation system messages must not carry tool_addition through natively: without
// the beta the API 400s "Input tag 'tool_addition' ... does not match any of the expected tags".
func TestToolChangeBlocks_StrippedWhereBetaUnsupported(t *testing.T) {
	for _, provider := range []schemas.ModelProvider{schemas.DeepSeek, schemas.Fireworks} {
		t.Run(string(provider)+"/tool_addition only", func(t *testing.T) {
			setMidConvSystemRecord(t, provider)
			out, err := ToAnthropicResponsesRequest(effortTestCtx(t), toolAdditionRequest(provider, false))
			if err != nil {
				t.Fatalf("ToAnthropicResponsesRequest: %v", err)
			}
			if n, _ := countToolChangeBlocks(out.Messages); n != 0 {
				t.Fatalf("%d tool-change blocks reached %s, which drops the beta", n, provider)
			}
			if last := out.Messages[len(out.Messages)-1]; last.Role == AnthropicMessageRoleSystem {
				t.Fatalf("empty system message left behind")
			}
		})
		t.Run(string(provider)+"/text kept natively", func(t *testing.T) {
			setMidConvSystemRecord(t, provider)
			out, err := ToAnthropicResponsesRequest(effortTestCtx(t), toolAdditionRequest(provider, true))
			if err != nil {
				t.Fatalf("ToAnthropicResponsesRequest: %v", err)
			}
			n, text := countToolChangeBlocks(out.Messages)
			if n != 0 {
				t.Fatalf("%d tool-change blocks reached %s", n, provider)
			}
			if text != 1 {
				t.Fatalf("system text blocks = %d, want 1 (native mid-conversation text lost)", text)
			}
		})
	}
}

// Anthropic direct forwards the beta, so the block must still reach it.
func TestToolChangeBlocks_KeptOnAnthropic(t *testing.T) {
	out, err := ToAnthropicResponsesRequest(effortTestCtx(t), toolAdditionRequest(schemas.Anthropic, false))
	if err != nil {
		t.Fatalf("ToAnthropicResponsesRequest: %v", err)
	}
	if n, _ := countToolChangeBlocks(out.Messages); n != 1 {
		t.Fatalf("tool-change blocks = %d, want 1 on Anthropic", n)
	}
}
