<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
  xmlns:atom="http://www.w3.org/2005/Atom" xmlns:dc="http://purl.org/dc/elements/1.1/"
  xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd">
  <xsl:output method="html" version="1.0" encoding="UTF-8" indent="yes" />
  <xsl:template match="/">
    <html xmlns="http://www.w3.org/1999/xhtml">
      <head>
        <meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1" />
        <title>Atom - <xsl:value-of select="/atom:feed/atom:title" /></title>
        <link rel="icon" href="./static/icon.webp" />
        <link href="/index.css" rel="stylesheet" type="text/css" />
        <style type="text/css">
          body{max-width:768px;margin:0 auto}section{margin:30px
          15px}hgroup{margin-bottom:2rem}a{text-decoration:none}
        </style>
      </head>
      <body>
        <xsl:apply-templates select="atom:feed" />
        <hr />
        <main xmlns="http://www.w3.org/1999/xhtml">
          <hgroup style="border-bottom:1px solid var(--lightgray);">
            <h2>Recent Items</h2>
            <p>
              <xsl:value-of select="atom:feed/atom:subtitle" />
            </p>
          </hgroup>
          <ul class="section-ul" xmlns="http://www.w3.org/1999/xhtml">
            <xsl:apply-templates select="atom:feed/atom:entry" />
          </ul>
        </main>
      </body>
    </html>
  </xsl:template>

  <xsl:template match="atom:feed">
    <header class="rss" xmlns="http://www.w3.org/1999/xhtml">
      <h1 class="article-title">
        <xsl:value-of select="atom:title" />
      </h1>
      <p>You have stumbled upon the <a
          href="https://www.ietf.org/rfc/rfc4287.txt" target="_blank"
          class="anchor-like"><span class="indicator-hook" />atom feed</a> of my working notes, as
        do to all paths of this digital garden. Much of these notes/writings are written for my own
        consumption, a sort of <a target="_blank"
          href="https://aarnphm.xyz/tags/evergreen">
          <span>evergreen</span>
        </a> notes. <br />If any of these doesn't make sense for
        you, it is probably because I didn't write it for you. <br /> 👋 you can reach out to me on <a
          href="https://twitter.com/aarnphm_" target="_blank">twitter</a> (Yep, I refused to call it
        call it X) </p>
      <a target="_blank">
        <xsl:attribute name="href">
          <xsl:value-of select="atom:link[@rel='alternate']/@href" />
        </xsl:attribute>main
        site &#x2192;</a>

      <p>There is also a <a href="/index.xml">RSS feed</a> of the site. They are <a
          href="https://news.ycombinator.com/item?id=26168493" target="_blank">semantically
          different but achieve the same thing.</a></p>

      <p>Visit <a href="https://aboutfeeds.com/">About
          Feeds</a> to get started with newsreaders and subscribing. It’s free. </p>

      <blockquote
        class="callout tip" data-callout="tip">
        <div class="callout-title" dir="auto">
          <div class="callout-icon" dir="auto"></div>
          <div class="callout-title-inner" dir="auto">
            <p dir="auto">subscribe</p>
          </div>
        </div>
        <div class="callout-content" dir="auto">
          <p dir="auto">On most slash-command supported interface, you can use the following <code>/feed
            subscribe <xsl:value-of select="atom:link[@rel='alternate']/@href" />/feed.xml</code></p>
        </div>
      </blockquote>
    </header>
  </xsl:template>

  <xsl:template match="atom:entry">
    <li class="section-li">
      <a target="_blank" data-list="true" class="note-link">
        <xsl:attribute name="href">
          <xsl:value-of select="atom:link/@href" />
        </xsl:attribute>
        <div class="note-grid">
          <div class="meta">
            <xsl:value-of select="atom:publishedTime" />
          </div>
          <div class="desc">
            <xsl:value-of select="atom:title" />
          </div>
          <menu class="tag-highlights">
            <xsl:for-each select="atom:category">
              <li class="tag">
                <xsl:value-of select="@term" />
              </li>
            </xsl:for-each>
          </menu>
        </div>
      </a>
    </li>
  </xsl:template>

</xsl:stylesheet>
