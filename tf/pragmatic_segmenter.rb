#!/usr/bin/env ruby
require "pragmatic_segmenter"

lang = ARGV[0]

begin
    while text = $stdin.gets
        ps = PragmaticSegmenter::Segmenter.new(text: text, language: lang)
        ps.segment.each do |sent|
            puts(sent)
        end
    end
rescue Interrupt => e
    # ignore
end
