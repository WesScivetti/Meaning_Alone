library(tidyverse)
library(ggtext)

deltas_no_syntax <- read_tsv("../outputs_summary/EleutherAI/pythia-12b/accuracy_summary_la_deltas.tsv")
deltas_syntax <- read_tsv("../outputs_summary/EleutherAI/pythia-12b/accuracy_summary_syntax_deltas.tsv")


deltas <- bind_rows(
  deltas_syntax %>%
    filter(test_type == "Avg_Test") %>%
    mutate(
      step = str_remove(revision, "step") %>%
        as.numeric()
    ) %>%
    group_by(step) %>%
    summarize(accuracy = mean(accuracy), delta_accuracy_zscale = mean(delta_accuracy_zscale)) %>%
    mutate(
      dataset = "PF Form"
    ),
  deltas_no_syntax %>%
    filter(world_knowledge_filter %in% c("aligned", "misaligned", "all") | str_detect(world_knowledge_filter, "domain")) %>%
    mutate(
      dataset = case_when(
        construction %in% c("blimp", "comps") ~ construction,
        construction %in% c("ewok") ~ glue::glue("EWoK\n({world_knowledge_filter})"),
        TRUE ~ glue::glue("PF Function\n({world_knowledge_filter})"),
      ),
      dataset = case_when(
        construction == "blimp" ~ "BLIMP",
        # construction == "ewok" ~ "EWoK",
        construction == "comps" ~ "COMPS",
        TRUE ~ dataset
      ),
      step = str_remove(revision, "step") %>%
        as.numeric()
    ) %>%
    filter(dataset != "PF Function\n(all)") %>% 
    group_by(dataset, step) %>%
    summarize(
      accuracy = mean(accuracy),
      delta_accuracy_zscale = mean(delta_accuracy_zscale)
    ) %>%
    ungroup() 
) 

deltas %>%
  widyr::pairwise_cor(dataset, step, delta_accuracy_zscale, method = "spearman") %>% 
  View()


deltas_no_syntax %>%
  filter(!construction %in% c("muchless", "nottomention", "nevermind"))

alph = 0.7
colooor = "black"

ewok_plausible <- deltas %>%
  filter(dataset %in% c("EWoK\n(domain_physical-relations)", "PF Function\n(aligned)")) %>%
  select(-accuracy) %>%
  pivot_wider(names_from = dataset, values_from = delta_accuracy_zscale) %>%
  janitor::clean_names() %>%
  # ggplot(aes(e_wo_k_domain_physical_relations, pf_function_aligned)) +
  ggplot(aes(pf_function_aligned, e_wo_k_domain_physical_relations)) +
  geom_point(alpha = alph, color = colooor) +
  geom_hline(yintercept = 0, linetype = "dashed", linewidth = 0.4, color = "grey50") +
  geom_vline(xintercept = 0, linetype = "dashed", linewidth = 0.4, color = "grey50") +
  # annotate("text", x = 1.0, y = 0.8, labels = "") +
  geom_richtext(x = 1.0, y = 0.8, label = "<i>&rho;</i> = 0.48") +
  # geom_smooth(method = "lm") +
  # scale_y_continuous(limits = c(-0.8, 1.8), breaks = scales::pretty_breaks()) +
  # scale_y_continuous(limits = c(-1.6, 1.6)) +
  # scale_x_continuous(limits = c(-1, 1)) +
  scale_y_continuous(limits = c(-1,1)) +
  scale_x_continuous(limits = c(-0.8, 1.8), breaks = scales::pretty_breaks()) +
  theme_classic(base_size = 16, base_family = "Times") +
  theme(
    axis.title.x = element_markdown(),
    axis.title.y = element_markdown()
  ) +
  labs(
    x = "&Delta;PF Function <span style='font-size: 14pt;'>(<i>plausible</i>)</span>",
    y = "&Delta;EWoK <span style='font-size: 14pt;'>(<i>Physical Relations</i>)</span>"
  ) 
  
# 363w 319h
# ggsave("ewok-plausible-corr.pdf", plot = ewok_plausible, width = 3.63, height = 3.19, device=cairo_pdf)

implausible_plausible <- deltas %>%
  filter(dataset %in% c("PF Function\n(misaligned)", "PF Function\n(aligned)")) %>%
  select(-accuracy) %>%
  pivot_wider(names_from = dataset, values_from = delta_accuracy_zscale) %>%
  janitor::clean_names() %>%
  # ggplot(aes(e_wo_k_domain_physical_relations, pf_function_aligned)) +
  ggplot(aes(pf_function_aligned, pf_function_misaligned)) +
  geom_point(alpha = alph, color = colooor) +
  geom_hline(yintercept = 0, linetype = "dashed", linewidth = 0.4, color = "grey50") +
  geom_vline(xintercept = 0, linetype = "dashed", linewidth = 0.4, color = "grey50") +
  geom_richtext(x = -0.2, y = 1.2, label = "<i>&rho;</i> = 0.47") +
  # geom_richtext(x = 1.0, y = 0.8, label = "<i>&rho;</i> = 0.48") +
  # geom_smooth(method = "lm") +
  # scale_y_continuous(limits = c(-0.8, 1.8), breaks = scales::pretty_breaks()) +
  # scale_y_continuous(limits = c(-1.6, 1.6)) +
  # scale_x_continuous(limits = c(-1, 1)) +
  scale_y_continuous(limits = c(-1,1.5)) +
  # scale_x_continuous(limits = c(-0.8, 1.8), breaks = scales::pretty_breaks()) +
  theme_classic(base_size = 16, base_family = "Times") +
  theme(
    axis.title.x = element_markdown(),
    axis.title.y = element_markdown()
  ) +
  labs(
    x = "&Delta;PF Function <span style='font-size: 14pt;'>(<i>plausible</i>)</span>",
    y = "&Delta;PF Function <span style='font-size: 14pt;'>(<i>implausible</i>)</span>"
  ) 

ggsave("~/projects/plots-for-projects/implausible-plausible-corr.pdf", plot = implausible_plausible, width = 3.63, height = 3.19, device=cairo_pdf)

form_plausible <- deltas %>%
  filter(dataset %in% c("PF Form", "PF Function\n(aligned)")) %>%
  select(-accuracy) %>%
  pivot_wider(names_from = dataset, values_from = delta_accuracy_zscale) %>%
  janitor::clean_names() %>%
  # ggplot(aes(e_wo_k_domain_physical_relations, pf_function_aligned)) +
  ggplot(aes(pf_function_aligned, pf_form)) +
  geom_point(alpha = alph, color = colooor) +
  geom_hline(yintercept = 0, linetype = "dashed", linewidth = 0.4, color = "grey50") +
  geom_vline(xintercept = 0, linetype = "dashed", linewidth = 0.4, color = "grey50") +
  # geom_richtext(x = -0.2, y = 1.2, label = "<i>&rho;</i> = 0.47") +
  geom_richtext(x = 1.0, y = 1.5, label = "<i>&rho;</i> = -0.01") +
  # geom_smooth(method = "lm") +
  # scale_y_continuous(limits = c(-0.8, 1.8), breaks = scales::pretty_breaks()) +
  # scale_y_continuous(limits = c(-1.6, 1.6)) +
  # scale_x_continuous(limits = c(-1, 1)) +
  scale_y_continuous(limits = c(-1,2)) +
  # scale_x_continuous(limits = c(-0.8, 1.8), breaks = scales::pretty_breaks()) +
  theme_classic(base_size = 16, base_family = "Times") +
  theme(
    axis.title.x = element_markdown(),
    axis.title.y = element_markdown()
  ) +
  labs(
    x = "&Delta;PF Function <span style='font-size: 14pt;'>(<i>plausible</i>)</span>",
    y = "&Delta;PF Form"
  ) 

#ggsave("form-plausible-corr.pdf", plot = form_plausible, width = 3.63, height = 3.19, device=cairo_pdf)


